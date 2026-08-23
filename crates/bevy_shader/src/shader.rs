use super::ShaderDefVal;
use alloc::borrow::Cow;
use bevy_asset::{io::Reader, Asset, AssetLoader, AssetPath, Handle, LoadContext};
use bevy_reflect::TypePath;
use bevy_utils::define_atomic_id;
use thiserror::Error;

/// Scans a WESL source for the modules it depends on.
///
/// `import a::b::c` always parses as the item `c` in module `a::b`, so what makes `c` a module
/// is how it is *used*, matching the upstream `wesl` compiler's `resolve_ty`:
///
/// * a bare use of `c` binds the item `c` inside module `a::b`, so only `a::b` is needed;
/// * a qualified use of `c::Item` reaches into the module `a::b::c`, so that module is needed too.
///
/// Nothing is guessed, so no dependency is ever fetched speculatively.
fn scan_wesl_imports(
    source: &str,
    self_module_path: &wesl::syntax::ModulePath,
) -> Vec<ShaderImport> {
    use wesl::syntax::{ImportContent, ModulePath, PathOrigin};

    /// Collects `(bound name, real name, module the item lives in)` per import. The two names
    /// differ under `as` renaming: use sites say the bound one, paths are built from the real one.
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
    fn push_required(required: &mut Vec<ModulePath>, path: ModulePath) {
        if !required.contains(&path) {
            required.push(path);
        }
    }

    let used = used_module_paths(&translation_unit);

    // The name a use site leads with. It always parses as the path's origin rather than a
    // component, because `super`, `package` and `self` are keywords, so a name an import
    // could have bound can only appear here.
    fn head(path: &ModulePath) -> Option<&str> {
        match &path.origin {
            PathOrigin::Package(package) => Some(package),
            _ => None,
        }
    }

    // A bare use binds the item inside the module it was imported from, so that module is the
    // dependency. A qualified use instead reaches *into* the item, making it a module in its
    // own right.
    for (bound, _, parent) in &paths {
        if !used.iter().any(|path| head(path) == Some(bound.as_str())) {
            push_required(&mut required, parent.clone());
        }
    }

    // A use site may also spell a module path out in full. If it leads with a name an import
    // bound, the rest of the path continues from that import — extended with the item's *real*
    // name, renaming having changed what the use site says and not what the module is called.
    // Otherwise the whole path is relative to this module.
    for path in &used {
        let imported = head(path).and_then(|head| {
            paths
                .iter()
                .find(|(bound, _, _)| bound == head)
                .map(|(_, real, parent)| (real, parent))
        });
        push_required(
            &mut required,
            match imported {
                Some((real, parent)) => {
                    let mut module = parent.clone();
                    module.push(real);
                    module.join(path.components.iter().cloned())
                }
                None => self_module_path.join_path(path),
            },
        );
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

/// Collects the module path written at each use site, e.g. the `color` of `color::TINT` or the
/// `super::file` of `super::file::item()`.
///
/// Only a [`TypeExpression`](wesl::syntax::TypeExpression) carries one, so this walks the same
/// nodes the upstream `wesl` compiler does in `resolve_ty`.
fn used_module_paths(
    translation_unit: &wesl::syntax::TranslationUnit,
) -> Vec<&wesl::syntax::ModulePath> {
    use wesl::syntax::{
        CompoundStatement, Expression, GlobalDeclaration, ModulePath, Statement, TypeExpression,
    };

    fn visit_type_expression<'a>(ty: &'a TypeExpression, out: &mut Vec<&'a ModulePath>) {
        if let Some(path) = &ty.path {
            out.push(path);
        }
        // Template arguments are themselves expressions, e.g. `array<pkg::mod::T, 4>`.
        for argument in ty.template_args.iter().flatten() {
            visit_expression(&argument.expression, out);
        }
    }

    fn visit_expression<'a>(expression: &'a Expression, out: &mut Vec<&'a ModulePath>) {
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

    fn visit_body<'a>(body: &'a CompoundStatement, out: &mut Vec<&'a ModulePath>) {
        for statement in &body.statements {
            visit_statement(statement, out);
        }
    }

    fn visit_statement<'a>(statement: &'a Statement, out: &mut Vec<&'a ModulePath>) {
        let mut expression = |expression| visit_expression(expression, out);
        match statement {
            Statement::Void
            | Statement::Break(_)
            | Statement::Continue(_)
            | Statement::Discard(_) => {}
            Statement::Compound(inner) => visit_body(inner, out),
            Statement::Assignment(inner) => {
                expression(&inner.lhs);
                expression(&inner.rhs);
            }
            Statement::Increment(inner) => expression(&inner.expression),
            Statement::Decrement(inner) => expression(&inner.expression),
            Statement::If(inner) => {
                visit_expression(&inner.if_clause.expression, out);
                visit_body(&inner.if_clause.body, out);
                for clause in &inner.else_if_clauses {
                    visit_expression(&clause.expression, out);
                    visit_body(&clause.body, out);
                }
                if let Some(clause) = &inner.else_clause {
                    visit_body(&clause.body, out);
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
                    visit_body(&clause.body, out);
                }
            }
            Statement::Loop(inner) => {
                visit_body(&inner.body, out);
                if let Some(continuing) = &inner.continuing {
                    visit_body(&continuing.body, out);
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
                visit_body(&inner.body, out);
            }
            Statement::While(inner) => {
                visit_expression(&inner.condition, out);
                visit_body(&inner.body, out);
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
                visit_body(&inner.body, &mut out);
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
    /// The module is still a dependency, whatever the path's origin or where it is written.
    ///
    /// Derived from the `inline super:: reference`, `inline package reference` and
    /// `uninitialized override` cases in `wesl-testsuite`'s `importCases.json`.
    #[test]
    fn inline_path_without_an_import_is_a_dependency() {
        let relative = Shader::from_wesl("fn main() { super::file1::bar(); }", "shaders/main.wesl");
        assert_eq!(relative.imports, vec![asset("/shaders/file1")]);

        let absolute = Shader::from_wesl(
            "fn main() { package::shaders::foo::bar(); }",
            "shaders/main.wesl",
        );
        assert_eq!(absolute.imports, vec![asset("/shaders/foo")]);

        let initializer =
            Shader::from_wesl("var a = package::shaders::file::b;", "shaders/main.wesl");
        assert_eq!(initializer.imports, vec![asset("/shaders/file")]);
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

        assert_eq!(shader.imports, vec![asset("/shaders/utils/color")]);
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

    /// A module may be nested under one reached by an import. The components written after the
    /// imported name are part of the module path and must not be dropped.
    #[test]
    fn module_nested_under_an_imported_name_is_a_dependency() {
        let shader = Shader::from_wesl(
            "import package::shaders::a::b;\nfn f() { b::c::D(); }",
            "shaders/main.wesl",
        );

        assert_eq!(shader.imports, vec![asset("/shaders/a/b/c")]);
    }

    /// A renamed import is used under its alias, so the alias is what decides.
    #[test]
    fn alias_is_what_decides_the_reading() {
        let shader = Shader::from_wesl(
            "import package::shaders::utils::color as c;\n\
             @fragment fn fragment() -> @location(0) vec4<f32> { return c::TINT; }",
            "shaders/root.wesl",
        );

        assert_eq!(shader.imports, vec![asset("/shaders/utils/color")]);
    }

    /// End-to-end over the real `assets/shaders/custom_material.wesl` used by the
    /// `shader_material` example — the exact file that reproduced bevyengine/bevy#25363.
    ///
    /// The asset loader turns every `AssetPath` import into a fetch, so this pins the whole
    /// set of requests. `COLOR_MULTIPLIER` is a `const`, not a file, and must never be one.
    #[test]
    fn real_example_asset_fetches_only_real_modules() {
        let source = include_str!("../../../assets/shaders/custom_material.wesl");
        let shader = Shader::from_wesl(source, "shaders/custom_material.wesl");

        let fetched: Vec<&ShaderImport> = shader
            .imports
            .iter()
            .filter(|import| matches!(import, ShaderImport::AssetPath(_)))
            .collect();

        assert_eq!(fetched, vec![&asset("/shaders/custom_material_import")]);
    }
}
