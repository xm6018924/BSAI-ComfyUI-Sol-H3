# Runtime implementation

The implementation files live here to keep the repository homepage short.
ComfyUI still loads the root `__init__.py`; node IDs, input/output schemas and
workflow locations are unchanged. This directory is added to the original
package search path, so qualified imports such as `<plugin>.sampling` keep the
same module identity. Do not import a second copy as `<plugin>.h3_t8.sampling`.

Vendored runtimes remain alongside the modules that use them. Standalone workers
also run from this directory. The three root TRT preparation/compile commands
remain as compatibility entrypoints. No models or engines belong in this folder.
