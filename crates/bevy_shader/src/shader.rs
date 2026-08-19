use super::ShaderDefVal;
use alloc::borrow::Cow;
use bevy_asset::{io::Reader, Asset, AssetLoader, AssetPath, Handle, LoadContext};
use bevy_reflect::TypePath;
use bevy_utils::define_atomic_id;
use thiserror::Error;

/// Scans a WESL source for the modules it depends on.
///
/// An import statement alone is ambiguous: `import a::b::c` cannot be told apart from the same
/// text where `c` names the module `a/b/c` — both parse as an
/// [`ImportContent::Item`](wesl::syntax::ImportContent::Item). The disambiguating signal is how
/// the name is *used*, not how it is imported, and this mirrors how the upstream `wesl` compiler
/// resolves it (`wesl::import::resolve_ty`):
///
/// * a bare use of `c` binds the item `c` inside module `a::b`, so only `a::b` is needed;
/// * a qualified use of `c::Item` reaches into the module `a::b::c`, so that module is needed too.
///
/// Imports are therefore only a name → path table; a module becomes a dependency when a
/// declaration actually refers to it. Nothing is guessed and no path that cannot exist is ever
/// produced, so no dependency is fetched speculatively.
fn scan_wesl_imports(
    source: &str,
    self_module_path: &wesl::syntax::ModulePath,
) -> Vec<ShaderImport> {
    use wesl::syntax::{ImportContent, ModulePath, PathOrigin};

    /// Collects, for each import, the name it binds locally, the item's real name, and the
    /// module path the item lives in. The bound name is what a use site refers to; the real name
    /// is what a path component must be built from, and the two differ under `as` renaming.
    fn leaves(
        content: &ImportContent,
        path: ModulePath,
        out: &mut Vec<(String, String, ModulePath)>,
    ) {
        match content {
            ImportContent::Item(item) => {
                let real = item.ident.to_string();
                let bound = item
                    .rename
                    .as_ref()
                    .map_or_else(|| real.clone(), ToString::to_string);
                out.push((bound, real, path));
            }
            ImportContent::Collection(collection) => {
                for import in collection {
                    let path = path.clone().join(import.path.iter().cloned());
                    leaves(&import.content, path, out);
                }
            }
        }
    }

    let Ok(translation_unit) = source.parse::<wesl::syntax::TranslationUnit>() else {
        return Vec::new();
    };

    let mut paths = Vec::new();
    for statement in &translation_unit.imports {
        match &statement.path {
            Some(import_path) => {
                let path = self_module_path.join_path(import_path);
                leaves(&statement.content, path, &mut paths);
            }
            None => {
                if let ImportContent::Collection(collection) = &statement.content {
                    for import in collection {
                        let mut components = import.path.iter().cloned();
                        if let Some(package) = components.next() {
                            let path =
                                ModulePath::new(PathOrigin::Package(package), components.collect());
                            leaves(&import.content, path, &mut paths);
                        }
                    }
                }
            }
        }
    }

    let mut required: Vec<ModulePath> = Vec::new();
    let mut push_required = |path: ModulePath| {
        if !required.contains(&path) {
            required.push(path);
        }
    };

    let used = used_type_expressions(&translation_unit);

    // The leading segment written at a use site. `color::TINT` parses with `color` as the path's
    // *origin*, not as a component, so both places have to be consulted.
    fn leading_segment(path: &ModulePath) -> Option<&str> {
        match &path.origin {
            PathOrigin::Package(package) => Some(package.as_str()),
            _ => path.components.first().map(String::as_str),
        }
    }

    // Renaming changes what a use site says, not what the module is called, so the path is
    // extended with the item's real name rather than the bound one.
    for (bound, real, parent) in &paths {
        let qualified = used.iter().any(|ty| {
            ty.path
                .as_ref()
                .and_then(leading_segment)
                .is_some_and(|first| first == bound)
        });
        let mut path = parent.clone();
        if qualified {
            path.push(real);
        }
        push_required(path);
    }

    // A use site may also spell a module path in full, with no import statement introducing it.
    // `ty.ident` names the used declaration, so the path itself is the module.
    for ty in &used {
        let Some(path) = &ty.path else { continue };
        // Paths rooted at an imported name were handled above.
        if leading_segment(path)
            .is_some_and(|first| paths.iter().any(|(bound, _, _)| bound == first))
        {
            continue;
        }
        push_required(self_module_path.join_path(path));
    }

    let mut imports: Vec<ShaderImport> = Vec::new();
    for path in &required {
        let path = match &path.origin {
            PathOrigin::Package(pkg) if pkg.contains('/') => Cow::Owned(ModulePath {
                origin: PathOrigin::Package(pkg.rsplit('/').next().unwrap().to_string()),
                components: path.components.clone(),
            }),
            _ => Cow::Borrowed(path),
        };
        let import = match &path.origin {
            PathOrigin::Absolute => {
                ShaderImport::AssetPath(format!("/{}", path.components.join("/")))
            }
            PathOrigin::Package(package) => ShaderImport::Custom(
                core::iter::once(package.as_str())
                    .chain(path.components.iter().map(String::as_str))
                    .collect::<Vec<_>>()
                    .join("::"),
            ),
            PathOrigin::Relative(_) => continue,
        };
        if !imports.contains(&import) {
            imports.push(import);
        }
    }
    imports
}

/// Collects every [`TypeExpression`](wesl::syntax::TypeExpression) reachable from a module's
/// declarations.
///
/// A `TypeExpression` is where a name is *used*, and its `path` is the qualification written at
/// that use site. This is the same information the upstream `wesl` compiler walks in
/// `resolve_ty`, and it covers both forms a dependency can take:
///
/// * `c::Item`, where `c` was introduced by an import — `path` is `Some(c)`;
/// * `super::file::item`, written inline with no import statement at all — `path` is the whole
///   qualification.
fn used_type_expressions(
    translation_unit: &wesl::syntax::TranslationUnit,
) -> Vec<&wesl::syntax::TypeExpression> {
    use wesl::syntax::{Expression, GlobalDeclaration, Statement, TypeExpression};

    fn visit_type_expression<'a>(ty: &'a TypeExpression, out: &mut Vec<&'a TypeExpression>) {
        out.push(ty);
        // Template arguments are themselves expressions, e.g. `array<pkg::mod::T, 4>`.
        for argument in ty.template_args.iter().flatten() {
            visit_expression(&argument.expression, out);
        }
    }

    fn visit_expression<'a>(expression: &'a Expression, out: &mut Vec<&'a TypeExpression>) {
        match expression {
            Expression::Literal(_) => {}
            Expression::Parenthesized(inner) => visit_expression(&inner.expression, out),
            Expression::NamedComponent(inner) => visit_expression(&inner.base, out),
            Expression::Indexing(inner) => {
                visit_expression(&inner.base, out);
                visit_expression(&inner.index, out);
            }
            Expression::Unary(inner) => visit_expression(&inner.operand, out),
            Expression::Binary(inner) => {
                visit_expression(&inner.left, out);
                visit_expression(&inner.right, out);
            }
            Expression::FunctionCall(inner) => {
                visit_type_expression(&inner.ty, out);
                for argument in &inner.arguments {
                    visit_expression(argument, out);
                }
            }
            Expression::TypeOrIdentifier(ty) => visit_type_expression(ty, out),
        }
    }

    fn visit_statement<'a>(statement: &'a Statement, out: &mut Vec<&'a TypeExpression>) {
        let mut expression = |expression| visit_expression(expression, out);
        match statement {
            Statement::Void
            | Statement::Break(_)
            | Statement::Continue(_)
            | Statement::Discard(_) => {}
            Statement::Compound(inner) => {
                for statement in &inner.statements {
                    visit_statement(statement, out);
                }
            }
            Statement::Assignment(inner) => {
                expression(&inner.lhs);
                expression(&inner.rhs);
            }
            Statement::Increment(inner) => expression(&inner.expression),
            Statement::Decrement(inner) => expression(&inner.expression),
            Statement::If(inner) => {
                visit_expression(&inner.if_clause.expression, out);
                for statement in &inner.if_clause.body.statements {
                    visit_statement(statement, out);
                }
                for clause in &inner.else_if_clauses {
                    visit_expression(&clause.expression, out);
                    for statement in &clause.body.statements {
                        visit_statement(statement, out);
                    }
                }
                if let Some(clause) = &inner.else_clause {
                    for statement in &clause.body.statements {
                        visit_statement(statement, out);
                    }
                }
            }
            Statement::Switch(inner) => {
                visit_expression(&inner.expression, out);
                for clause in &inner.clauses {
                    for selector in &clause.case_selectors {
                        if let wesl::syntax::CaseSelector::Expression(expression) = selector {
                            visit_expression(expression, out);
                        }
                    }
                    for statement in &clause.body.statements {
                        visit_statement(statement, out);
                    }
                }
            }
            Statement::Loop(inner) => {
                for statement in &inner.body.statements {
                    visit_statement(statement, out);
                }
                if let Some(continuing) = &inner.continuing {
                    for statement in &continuing.body.statements {
                        visit_statement(statement, out);
                    }
                    if let Some(break_if) = &continuing.break_if {
                        visit_expression(&break_if.expression, out);
                    }
                }
            }
            Statement::For(inner) => {
                if let Some(statement) = &inner.initializer {
                    visit_statement(statement, out);
                }
                if let Some(condition) = &inner.condition {
                    visit_expression(condition, out);
                }
                if let Some(statement) = &inner.update {
                    visit_statement(statement, out);
                }
                for statement in &inner.body.statements {
                    visit_statement(statement, out);
                }
            }
            Statement::While(inner) => {
                visit_expression(&inner.condition, out);
                for statement in &inner.body.statements {
                    visit_statement(statement, out);
                }
            }
            Statement::Return(inner) => {
                if let Some(expression) = &inner.expression {
                    visit_expression(expression, out);
                }
            }
            Statement::FunctionCall(inner) => {
                visit_type_expression(&inner.call.ty, out);
                for argument in &inner.call.arguments {
                    visit_expression(argument, out);
                }
            }
            Statement::ConstAssert(inner) => visit_expression(&inner.expression, out),
            Statement::Declaration(inner) => {
                if let Some(ty) = &inner.ty {
                    visit_type_expression(ty, out);
                }
                if let Some(expression) = &inner.initializer {
                    visit_expression(expression, out);
                }
            }
        }
    }

    let mut out = Vec::new();
    for declaration in &translation_unit.global_declarations {
        match &**declaration {
            GlobalDeclaration::Declaration(inner) => {
                if let Some(ty) = &inner.ty {
                    visit_type_expression(ty, &mut out);
                }
                if let Some(expression) = &inner.initializer {
                    visit_expression(expression, &mut out);
                }
            }
            GlobalDeclaration::TypeAlias(inner) => visit_type_expression(&inner.ty, &mut out),
            GlobalDeclaration::Struct(inner) => {
                for member in &inner.members {
                    visit_type_expression(&member.ty, &mut out);
                }
            }
            GlobalDeclaration::Function(inner) => {
                for parameter in &inner.parameters {
                    visit_type_expression(&parameter.ty, &mut out);
                }
                if let Some(ty) = &inner.return_type {
                    visit_type_expression(ty, &mut out);
                }
                for statement in &inner.body.statements {
                    visit_statement(statement, &mut out);
                }
            }
            GlobalDeclaration::ConstAssert(inner) => visit_expression(&inner.expression, &mut out),
            _ => {}
        }
    }
    out
}

define_atomic_id!(ShaderId);

/// Describes whether or not to perform runtime checks on shaders.
/// Runtime checks can be enabled for safety at the cost of speed.
/// By default no runtime checks will be performed.
///
/// # Panics
/// Because no runtime checks are performed for spirv,
/// enabling `ValidateShader` for spirv will cause a panic
#[derive(Clone, Debug, Default)]
pub enum ValidateShader {
    #[default]
    /// No runtime checks for soundness (e.g. bound checking) are performed.
    ///
    /// This is suitable for trusted shaders, written by your program or dependencies you trust.
    Disabled,
    /// Enable's runtime checks for soundness (e.g. bound checking).
    ///
    /// While this can have a meaningful impact on performance,
    /// this setting should *always* be enabled when loading untrusted shaders.
    /// This might occur if you are creating a shader playground, running user-generated shaders
    /// (as in `VRChat`), or writing a web browser in Bevy.
    Enabled,
}

/// An "unprocessed" shader. It can contain imports and conditional
/// compilation attributes.
#[derive(Asset, TypePath, Debug, Clone)]
pub struct Shader {
    /// The asset path of the shader.
    pub path: String,
    /// The raw source code of the shader.
    pub source: Source,
    /// The path from which this shader can be imported by other shaders.
    pub import_path: ShaderImport,
    /// The import paths this shader depends on.
    pub imports: Vec<ShaderImport>,
    /// Any shader defs that should be included when this module is used.
    pub shader_defs: Vec<ShaderDefVal>,
    /// Strong handles to this shader's dependencies, to prevent them
    /// from being immediately dropped if this shader is the only user.
    pub file_dependencies: Vec<Handle<Shader>>,
    /// Enable or disable runtime shader validation, trading safety against speed.
    ///
    /// Please read the [`ValidateShader`] docs for a discussion of the tradeoffs involved.
    pub validate_shader: ValidateShader,
}

impl Shader {
    /// Creates a new WGSL shader.
    pub fn from_wgsl(source: impl Into<Cow<'static, str>>, path: impl Into<String>) -> Shader {
        let source = source.into();
        let path = path.into();
        Shader {
            import_path: ShaderImport::AssetPath(path.clone()),
            path,
            imports: Vec::new(),
            source: Source::Wgsl(source),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }

    /// Creates a new WGSL shader with some given shader defs.
    pub fn from_wgsl_with_defs(
        source: impl Into<Cow<'static, str>>,
        path: impl Into<String>,
        shader_defs: Vec<ShaderDefVal>,
    ) -> Shader {
        Self {
            shader_defs,
            ..Self::from_wgsl(source, path)
        }
    }

    /// Creates a new SPIR-V shader.
    pub fn from_spirv(source: impl Into<Cow<'static, [u8]>>, path: impl Into<String>) -> Shader {
        let path = path.into();
        Shader {
            path: path.clone(),
            imports: Vec::new(),
            import_path: ShaderImport::AssetPath(path),
            source: Source::SpirV(source.into()),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }

    /// Creates a new Wesl shader.
    pub fn from_wesl(source: impl Into<Cow<'static, str>>, path: impl Into<String>) -> Shader {
        let source = source.into();
        let path = path.into();

        let import_path = match path.strip_prefix("embedded://") {
            Some(embedded_path) => ShaderImport::Custom(
                std::path::Path::new(embedded_path)
                    .with_extension("")
                    .to_string_lossy()
                    .split('/')
                    .filter(|component| !component.is_empty())
                    .collect::<Vec<_>>()
                    .join("::"),
            ),
            None => {
                // Create the shader import path - always starting with "/"
                let shader_path = std::path::Path::new("/").join(&path);

                // Convert to a string with forward slashes and without extension
                let import_path_str = shader_path
                    .with_extension("")
                    .to_string_lossy()
                    .replace('\\', "/");

                ShaderImport::AssetPath(import_path_str.to_string())
            }
        };

        let imports = crate::shader_cache::wesl_module_path(&import_path)
            .map(|module_path| scan_wesl_imports(&source, &module_path))
            .unwrap_or_default();

        Shader {
            path,
            imports,
            import_path,
            source: Source::Wesl(source),
            shader_defs: Default::default(),
            file_dependencies: Default::default(),
            validate_shader: ValidateShader::Disabled,
        }
    }
}

/// Raw shader source code.
#[expect(missing_docs, reason = "The variants are self-explanatory.")]
#[derive(Debug, Clone)]
pub enum Source {
    Wgsl(Cow<'static, str>),
    Wesl(Cow<'static, str>),
    SpirV(Cow<'static, [u8]>),
    // TODO: consider the following
    // PrecompiledSpirVMacros(HashMap<HashSet<String>, Vec<u32>>)
    // NagaModule(Module) ... Module impls Serialize/Deserialize
}

impl Source {
    /// The underlying source code string, unless it is SPIR-V.
    pub fn as_str(&self) -> &str {
        match self {
            Source::Wgsl(s) | Source::Wesl(s) => s,
            Source::SpirV(_) => panic!("spirv not yet implemented"),
        }
    }
}

/// The [`AssetLoader`] responsible for loading unprocessed shader assets.
#[derive(Default, TypePath)]
pub struct ShaderLoader;

/// An error encountered while loading a shader's source.
#[non_exhaustive]
#[derive(Debug, Error)]
#[expect(missing_docs, reason = "The variants are self-explanatory.")]
pub enum ShaderLoaderError {
    #[error("Could not load shader: {0}")]
    Io(#[from] std::io::Error),
    #[error("Could not parse shader: {0}")]
    Parse(#[from] alloc::string::FromUtf8Error),
}

/// Settings for loading shaders.
#[derive(serde::Serialize, serde::Deserialize, Debug, Default)]
pub struct ShaderSettings {
    /// The shader defs to apply when this shader is loaded.
    pub shader_defs: Vec<ShaderDefVal>,
}

impl AssetLoader for ShaderLoader {
    type Asset = Shader;
    type Settings = ShaderSettings;
    type Error = ShaderLoaderError;
    async fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Shader, Self::Error> {
        let ext = load_context
            .path()
            .path()
            .extension()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let path = load_context.path().to_string();
        // On windows, the path will inconsistently use \ or /.
        // TODO: remove this once AssetPath forces cross-platform "slash" consistency. See #10511
        let path = path.replace(std::path::MAIN_SEPARATOR, "/");
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).await?;
        if ext != "wesl" && !settings.shader_defs.is_empty() {
            tracing::warn!(
                "Tried to load a non-wesl shader with shader defs, this isn't supported: \
                    The shader defs will be ignored."
            );
        }
        let mut shader = match ext.as_str() {
            "spv" => Shader::from_spirv(bytes, load_context.path().path().to_string_lossy()),
            "wgsl" => Shader::from_wgsl(String::from_utf8(bytes)?, path),
            "wesl" => {
                let mut shader = Shader::from_wesl(String::from_utf8(bytes)?, path);
                shader.shader_defs = settings.shader_defs.clone();
                shader
            }
            _ => panic!("unhandled extension: {ext}"),
        };

        // collect and store file dependencies
        match ext.as_str() {
            "wesl" => {
                // Loading through the asset server means a module shared by several importers is
                // fetched once, rather than re-read per importer as an existence check would.
                let dependencies: Vec<String> = shader
                    .imports
                    .iter()
                    .filter_map(|import| match import {
                        ShaderImport::AssetPath(asset_path) => {
                            Some(format!("{}.{ext}", asset_path.trim_start_matches('/')))
                        }
                        ShaderImport::Custom(_) => None,
                    })
                    .collect();
                for file_path in dependencies {
                    shader
                        .file_dependencies
                        .push(load_context.load(AssetPath::from(file_path)));
                }
            }
            _ => {
                for import in &shader.imports {
                    if let ShaderImport::AssetPath(asset_path) = import {
                        shader.file_dependencies.push(load_context.load(asset_path));
                    }
                }
            }
        }
        Ok(shader)
    }

    fn extensions(&self) -> &[&str] {
        &["spv", "wgsl", "wesl"]
    }
}

/// A shader import, described as either an asset path or an import path.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum ShaderImport {
    /// An asset path to a shader.
    AssetPath(String),
    /// An import path from which a shader may be imported.
    Custom(String),
}

/// A reference to a shader asset.
#[derive(Default)]
pub enum ShaderRef {
    /// Use the "default" shader for the current context.
    #[default]
    Default,
    /// A handle to a shader stored in the [`Assets<Shader>`](bevy_asset::Assets) resource.
    Handle(Handle<Shader>),
    /// An asset path leading to a shader.
    Path(AssetPath<'static>),
}

impl From<Handle<Shader>> for ShaderRef {
    fn from(handle: Handle<Shader>) -> Self {
        Self::Handle(handle)
    }
}

impl From<AssetPath<'static>> for ShaderRef {
    fn from(path: AssetPath<'static>) -> Self {
        Self::Path(path)
    }
}

impl From<&'static str> for ShaderRef {
    fn from(path: &'static str) -> Self {
        Self::Path(AssetPath::from(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn asset(path: &str) -> ShaderImport {
        ShaderImport::AssetPath(path.to_string())
    }

    /// A module path may be written inline in a declaration, with no import statement at all.
    /// The module is still a dependency.
    ///
    /// Derived from the `inline super:: reference`, `inline package reference` and
    /// `uninitialized override` cases in `wesl-testsuite`'s `importCases.json`.
    #[test]
    fn inline_path_without_an_import_is_a_dependency() {
        let shader = Shader::from_wesl("fn main() { super::file1::bar(); }", "shaders/main.wesl");

        assert_eq!(shader.imports, vec![asset("/shaders/file1")]);
    }

    #[test]
    fn inline_package_path_is_a_dependency() {
        let shader = Shader::from_wesl(
            "fn main() { package::shaders::foo::bar(); }",
            "shaders/main.wesl",
        );

        assert_eq!(shader.imports, vec![asset("/shaders/foo")]);
    }

    #[test]
    fn inline_path_in_a_global_initializer_is_a_dependency() {
        let shader = Shader::from_wesl("var a = package::shaders::file::b;", "shaders/main.wesl");

        assert_eq!(shader.imports, vec![asset("/shaders/file")]);
    }

    /// A bare use of an imported name binds a declaration in the parent module, so only the
    /// parent is a dependency. The item's name must never become a module path.
    ///
    /// Regression test for bevyengine/bevy#25363, where `COLOR_MULTIPLIER` was scanned as a
    /// module and fetched, producing a spurious 404 on the web.
    #[test]
    fn bare_use_depends_only_on_the_parent_module() {
        let shader = Shader::from_wesl(
            "import super::custom_material_import::COLOR_MULTIPLIER;\n\
             @fragment fn fragment() -> @location(0) vec4<f32> { return COLOR_MULTIPLIER; }",
            "shaders/custom_material.wesl",
        );

        assert_eq!(
            shader.imports,
            vec![asset("/shaders/custom_material_import")]
        );
    }

    /// A qualified use reaches into a module, so the nested module is the dependency. Dropping it
    /// would silently break projects that organise shaders into subdirectories.
    #[test]
    fn qualified_use_depends_on_the_nested_module() {
        let shader = Shader::from_wesl(
            "import package::shaders::utils::color;\n\
             @fragment fn fragment() -> @location(0) vec4<f32> { return color::TINT; }",
            "shaders/root.wesl",
        );

        assert!(
            shader.imports.contains(&asset("/shaders/utils/color")),
            "nested module must be a dependency, got {:?}",
            shader.imports
        );
    }

    /// Importing an item from a nested module needs both: the module is reached by the qualified
    /// path in the import, and the item is a declaration inside it.
    #[test]
    fn item_from_nested_module_depends_on_that_module() {
        let shader = Shader::from_wesl(
            "import package::shaders::utils::color::TINT;\n\
             @fragment fn fragment() -> @location(0) vec4<f32> { return TINT; }",
            "shaders/root.wesl",
        );

        assert_eq!(shader.imports, vec![asset("/shaders/utils/color")]);
    }

    /// A renamed import is used under its alias, so the alias is what decides.
    #[test]
    fn alias_is_what_decides_the_reading() {
        let shader = Shader::from_wesl(
            "import package::shaders::utils::color as c;\n\
             @fragment fn fragment() -> @location(0) vec4<f32> { return c::TINT; }",
            "shaders/root.wesl",
        );

        assert!(
            shader.imports.contains(&asset("/shaders/utils/color")),
            "alias used qualified must reach the nested module, got {:?}",
            shader.imports
        );
    }
}
