"""Official installation audit and guarded regular Topaz worker orchestration."""
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .topaz_contract import (OfficialTopaz, DELIVERY_OUTPUT_PROFILES, REGULAR_OUTPUT_PROFILES,
    REGULAR_PARAMETERS, regular_filter, interpolation_filter)
from .topaz_media import file_identity


TOPAZ_STARTUP_FREE_RAM_BYTES = 16 * 1024**3


# These are read-only installation hints, not an installer contract.  Topaz
# keeps the program beside its signed binaries and the model catalog under
# ProgramData on Windows.  Keeping the candidates here lets the Director show
# a useful preflight result without asking a beginner to copy three paths from
# Explorer.  Explicit node paths still win, and no executable or license is
# opened by discovery.
_DEFAULT_TOPAZ_INSTALLS = (
    Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'Topaz Labs LLC' / 'Topaz Video AI',
    Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'Topaz Labs LLC' / 'Topaz Video',
)
_DEFAULT_TOPAZ_MODEL_ROOTS = (
    Path(os.environ.get('ProgramData', r'C:\ProgramData')) / 'Topaz Labs LLC' / 'Topaz Video AI' / 'models',
    Path(os.environ.get('ProgramData', r'C:\ProgramData')) / 'Topaz Labs LLC' / 'Topaz Video' / 'models',
)


def discover_official_installations(*, installs=None, model_roots=None):
    """Return local Topaz candidates using filesystem metadata only.

    Discovery is deliberately separate from :class:`OfficialTopaz`: it does
    not launch ``Topaz Video AI.exe``, invoke FFmpeg, inspect licensing state,
    or download models.  The result is suitable for a preflight UI and for the
    optional auto-discovery path of the environment node.  A candidate is
    ``ready`` only when the signed-runtime inputs and at least one JSON model
    definition are present; actual signatures, filter options and model loads
    remain the environment node's explicit audit.
    """
    install_paths = tuple(Path(value) for value in (installs or _DEFAULT_TOPAZ_INSTALLS))
    root_paths = tuple(Path(value) for value in (model_roots or _DEFAULT_TOPAZ_MODEL_ROOTS))
    rows = []
    seen = set()
    for install in install_paths:
        install = install.expanduser()
        key = str(install).lower()
        if key in seen:
            continue
        seen.add(key)
        # Prefer a model root whose product name matches the installation, then
        # fall back to any existing official catalog (older installs use a
        # different product-directory name).
        definitions = next((root for root in root_paths
                            if root.is_dir() and root.parent.name.lower() in install.name.lower()), None)
        if definitions is None:
            definitions = next((root for root in root_paths if root.is_dir()), None)
        executables = {name: (install / name).is_file()
                       for name in ('Topaz Video AI.exe', 'ffmpeg.exe', 'ffprobe.exe')}
        definition_count = 0
        weight_count = 0
        if definitions is not None:
            try:
                definition_count = sum(1 for path in definitions.glob('*.json') if path.is_file())
                weight_count = sum(1 for path in definitions.iterdir()
                                   if path.is_file() and path.suffix.lower() in ('.tz', '.tz3'))
            except OSError:
                definition_count = weight_count = 0
        ready = all(executables.values()) and definitions is not None and definition_count > 0
        rows.append({
            'install': str(install),
            'definitions': str(definitions) if definitions is not None else None,
            'data': str(definitions) if definitions is not None else None,
            'executables': executables,
            'model_definition_count': definition_count,
            'model_weight_count': weight_count,
            'status': 'ready' if ready else 'incomplete',
            'inference_executed': False,
            'license_or_login_read': False,
        })
    return sorted(rows, key=lambda row: (row['status'] != 'ready', row['install'].lower()))


def discover_official_installation(**kwargs):
    """Return the best local candidate, or ``None`` when none is installed."""
    return next((row for row in discover_official_installations(**kwargs)
                 if row['status'] == 'ready'), None)


def _run_readonly(command, runtime, *, timeout=30, environment=None):
    result = subprocess.run(command, cwd=runtime.install,
        env=environment or runtime.child_environment(os.environ), capture_output=True,
        timeout=timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), shell=False)
    if result.returncode:
        raise RuntimeError('Official runtime inspection failed, exit code ' + str(result.returncode)
            + ': ' + result.stderr.decode('utf8', 'replace')[-2400:])
    return (result.stdout + result.stderr).decode('utf8', 'replace')


def validate_signatures(signatures, paths):
    if (not isinstance(signatures, list) or len(signatures) != len(paths)
            or any(not isinstance(s, dict) or s.get('path') != str(p)
                or s.get('status') != 'Valid'
                or 'O=Topaz Labs LLC' not in (s.get('signer') or '')
                for s, p in zip(signatures, paths))):
        raise RuntimeError('All selected executables must have valid official Topaz Labs signatures')


def audit_installation(runtime: OfficialTopaz):
    if os.name != 'nt':
        raise RuntimeError('This official Topaz integration currently requires Windows')
    files = [runtime.executable(name) for name in ('Topaz Video.exe', 'ffmpeg.exe', 'ffprobe.exe')]
    identities = [file_identity(path) for path in files]
    # Fixed PowerShell code; paths travel as JSON in a child-only variable, never
    # interpolated into shell text. Do not inspect license/login files.
    code = ("$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; [Console]::OutputEncoding=[Text.Encoding]::UTF8; "
        "Import-Module Microsoft.PowerShell.Security -ErrorAction Stop; "
        "$paths=ConvertFrom-Json $env:T8_TOPAZ_AUDIT_PATHS; $rows=foreach($path in $paths) { "
        "$s=Get-AuthenticodeSignature -LiteralPath $path; "
        "[pscustomobject]@{path=$path;status=[string]$s.Status;signer=$s.SignerCertificate.Subject;"
        "version=(Get-Item -LiteralPath $path).VersionInfo.FileVersion} }; ConvertTo-Json -InputObject @($rows) -Compress")
    ps = Path(os.environ['SystemRoot']) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    env = runtime.child_environment(os.environ)
    # Do not inherit PowerShell7 module paths into Windows PowerShell5.1.
    # Its built-in security module performs signature checks; no policy bypass.
    env = {k: v for k, v in env.items() if k.upper() != 'PSMODULEPATH'}
    env['T8_TOPAZ_AUDIT_PATHS'] = json.dumps([str(p) for p in files])
    signatures = json.loads(_run_readonly([str(ps), '-NoProfile', '-NonInteractive', '-EncodedCommand',
        base64.b64encode(code.encode('utf-16-le')).decode('ascii')], runtime, environment=env).lstrip('\ufeff'))
    validate_signatures(signatures, files)
    if [file_identity(path) for path in files] != identities:
        raise RuntimeError('Installation changed during audit')
    help_text = _run_readonly([str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-h', 'filter=tvai_up'], runtime)
    options = sorted(set(re.findall(r'^\s+(\w+)\s+<', help_text, re.M)))
    required = {'model', 'scale', 'w', 'h', 'device', 'instances', 'download', 'vram'}
    if 'Filter tvai_up' not in help_text or not required.issubset(options):
        raise RuntimeError('This official FFmpeg lacks the required tvai_up interface')
    try:
        fi_help = _run_readonly([str(runtime.executable('ffmpeg.exe')), '-hide_banner', '-h', 'filter=tvai_fi'], runtime)
    except RuntimeError:
        fi_help = ''
    fi_options = sorted(set(re.findall(r'^\s+(\w+)\s+<', fi_help, re.M)))
    fi_required = {'model', 'device', 'instances', 'download', 'vram', 'slowmo', 'rdt', 'fps'}
    fi_available = 'Filter tvai_fi' in fi_help and fi_required.issubset(fi_options)
    return {'status': 'official_interface_verified_not_model_inference',
        'executables': identities, 'signatures': signatures, 'tvai_up_options': options,
        'tvai_fi_options': fi_options, 'tvai_fi_available': fi_available,
        'neuroserver_directory_present': (runtime.install / 'neuroserver').is_dir(),
        'model_catalog': model_catalog(runtime, options),
        'downloads': False, 'license_or_login_read': False}


def _declared_weight_patterns(definition, scale):
    """Return safe filename regexes from the selected scale's official backend nets."""
    templates = set()

    def collect(value):
        if not isinstance(value, dict):
            return
        nets = value.get('nets', [])
        if isinstance(nets, list):
            templates.update(net for net in nets if isinstance(net, str))
        specs = value.get('spec', {})
        if isinstance(specs, dict):
            for spec in specs.values():
                collect(spec)

    backends = definition.get('backends', {})
    if isinstance(backends, dict):
        for backend in backends.values():
            if not isinstance(backend, dict):
                continue
            scales = backend.get('scales', {})
            if isinstance(scales, dict):
                collect(scales.get(str(scale), scales.get(scale)))
    patterns = []
    for template in sorted(templates):
        if (Path(template).name != template or not template.endswith('.tz')
                or set(re.findall(r'\[[A-Z][A-Z0-9_]*\]', template))
                - {'[H]', '[W]', '[S]', '[C]', '[R]'}):
            raise ValueError('Unsupported model weight template in definition')
        expression = re.escape(template[:-3])
        expression = expression.replace(re.escape('[H]'), r'\d+')
        expression = expression.replace(re.escape('[W]'), r'\d+')
        expression = expression.replace(re.escape('[S]'), str(scale))
        # Official TensorRT catalogs use [C]/[R] for capability and runtime
        # build variants (for example rt809-10800). They are numeric slots,
        # not arbitrary path text, and must remain bound to the selected
        # model's filename template.
        expression = expression.replace(re.escape('[C]'), r'\d+')
        expression = expression.replace(re.escape('[R]'), r'\d+')
        patterns.append(re.compile(expression + r'\.tz3?\Z'))
    return patterns


def _candidate_weights(runtime, definition, scale, files=None):
    short, version = definition.get('shortName'), definition.get('version')
    if (not isinstance(short, str) or not re.fullmatch('[a-z0-9-]+', short)
            or type(version) not in (str, int) or not re.fullmatch('[0-9]{1,6}', str(version))):
        raise ValueError('Unsupported model definition naming contract')
    prefix = f'{short}-v{version}-'
    files = runtime.data.iterdir() if files is None else files
    patterns = _declared_weight_patterns(definition, scale)
    weights = [p for p in files if p.is_file() and p.name.startswith(prefix)
        and p.suffix in ('.tz', '.tz3')
        and (any(pattern.fullmatch(p.name[len(prefix):]) for pattern in patterns)
            if patterns else f'-{scale}x-' in p.name)]
    if any(p.resolve(strict=True).parent != runtime.data for p in weights):
        raise ValueError('Candidate model weights leave the selected data directory')
    return sorted(weights)


def model_catalog(runtime, runtime_options):
    """Read-only directory inventory, never a license or execution readiness probe."""
    models, skipped = [], []
    files = list(runtime.data.iterdir())
    options = set(runtime_options)
    for entry in sorted(runtime.definitions.glob('*.json')):
        try:
            path, definition = runtime.model(entry.stem)
            row = {'id': entry.stem, 'definition_path': str(path),
                   'execution_verified': False, 'enabled_in_definition': definition.get('enabled'),
                   'definition_name': definition.get('displayName') or definition.get('name') or definition.get('shortName') or entry.stem}
            if definition.get('isNeuroserverModel'):
                row.update(route='neuroserver', status='separate_neuroserver_qualification_required')
            elif definition.get('changesFPS') and definition.get('modelType') == 2:
                row.update(route='tvai_fi', status='frame_interpolation_definition_discovered')
            elif definition.get('changesFPS') or definition.get('modelType') != 1:
                row.update(route='not_offered_by_regular_upscale',
                           status='fps_auxiliary_or_unclassified_definition',
                           declared_model_type=definition.get('modelType'))
            else:
                row.update(route='tvai_up', status='definition_discovered', scales={})
                for scale in (1, 2, 4):
                    weights = _candidate_weights(runtime, definition, scale, files)
                    row['scales'][str(scale)] = {
                        'status': 'candidate_files_present_unverified' if weights else 'missing_candidate_weights',
                        'candidate_files': [{'name': p.name, 'bytes': p.stat().st_size} for p in weights]}
                parameters = {}
                for item in definition.get('parameters', []):
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get('name', '')).lower()
                    if name in REGULAR_PARAMETERS & options and name not in ('estimate', 'kcolor', 'blend'):
                        parameters[name] = {key: item[key] for key in
                            ('min', 'max', 'default', 'guiName') if key in item}
                        parameters[name]['source'] = 'selected_model_definition'
                for name, maximum in (('blend', 1), ('estimate', 100), ('kcolor', 1)):
                    if name in options:
                        parameters[name] = {'min': 0, 'max': maximum, 'source': 'runtime_option',
                                            'integer_only': name != 'blend'}
                row['parameters'] = parameters
            models.append(row)
        except (OSError, ValueError, TypeError) as error:
            skipped.append({'id': entry.stem, 'reason': str(error)})
    return {'models': models, 'unreadable_or_nonmodel_definitions': skipped,
            'inference_executed': False, 'weight_hashes_computed': False,
            'scope': 'Definitions and scale-specific candidate filenames only; engine loading, license and output are not verified. Exact weight identities are checked by enhancement execution.'}


def model_evidence(runtime, model_id, scale):
    path, definition = runtime.model(model_id)
    if definition.get('isNeuroserverModel'):
        raise ValueError('Formal Starlight requires a separately verified Neuroserver route')
    weights = _candidate_weights(runtime, definition, scale)
    if not weights:
        raise RuntimeError(f'{model_id}: no installed {scale}x weight candidates. Download the selected model through official Topaz first; automatic downloading is disabled.')
    return {'definition': file_identity(path), 'scale': scale,
        'candidate_weights': [file_identity(p) for p in sorted(weights)],
        'status': 'candidate_files_present_actual_engine_load_still_required'}


def dimension_model_evidence(runtime, model_id):
    """Bind available variants; engine-selected variant is not assumed from size."""
    path, definition = runtime.model(model_id)
    if definition.get('isNeuroserverModel') or definition.get('changesFPS') or definition.get('modelType', 1) != 1:
        raise ValueError('Custom dimensions require a regular enhancement model')
    files = list(runtime.data.iterdir())
    by_scale = {scale: _candidate_weights(runtime, definition, scale, files) for scale in (1, 2, 4)}
    weights = sorted({p for candidates in by_scale.values() for p in candidates})
    if not weights:
        raise RuntimeError(f'{model_id}: no installed weight candidates; prepare the model through official Topaz first')
    return {'definition': file_identity(path), 'scale': 'engine_auto_dimensions',
        'candidate_weights': [file_identity(p) for p in weights],
        'missing_candidate_scales': [scale for scale, candidates in by_scale.items() if not candidates],
        'actual_engine_variant_verified': False,
        'status': 'candidate_files_present_actual_engine_load_still_required'}


def interpolation_model_evidence(runtime, model_id):
    path, definition = runtime.model(model_id)
    if not definition.get('changesFPS') or definition.get('modelType') != 2:
        raise ValueError('Selected model is not a regular Topaz frame interpolator')
    short, version = definition.get('shortName'), definition.get('version')
    if (not isinstance(short, str) or not re.fullmatch('[a-z0-9-]+', short)
            or type(version) not in (str, int) or not re.fullmatch('[0-9]{1,6}', str(version))):
        raise ValueError('Unsupported interpolation model definition naming contract')
    prefix = f'{short}-v{version}-'
    weights = sorted(p for p in runtime.data.iterdir() if p.is_file()
        and p.name.startswith(prefix) and p.suffix in ('.tz', '.tz3'))
    if not weights:
        raise RuntimeError(f'{model_id}: no installed interpolation weights. Download this model through official Topaz first; automatic downloading is disabled.')
    return {'definition': file_identity(path),
        'candidate_weights': [file_identity(p) for p in weights],
        'status': 'candidate_files_present_actual_engine_load_still_required'}


def validate_publication(job, spec, report):
    if report.get('status') != 'media_audit_pass_human_pending':
        raise RuntimeError('Worker did not produce a verified enhancement publication')
    if report.get('source') != spec['source'] or file_identity(spec['source']['path']) != spec['source']:
        raise RuntimeError('Source identity changed before publication')
    expected = report.get('output', {})
    candidate = Path(expected.get('path', '')).resolve(strict=True)
    profile = spec.get('settings', {}).get('output_profile', 'delivery_h264')
    names = ('enhanced.mp4', 'enhanced.mkv') if profile in DELIVERY_OUTPUT_PROFILES else ('enhanced.mov', 'enhanced.mkv')
    if candidate.parent != Path(job).resolve(strict=True) or candidate.name not in names:
        raise RuntimeError('Output publication must be the completed task-owned master')
    if file_identity(candidate) != expected:
        raise RuntimeError('Output identity changed before publication')
    return candidate


def record_topaz_startup_sample(reader, guard, log):
    """Record the real device snapshot without imposing a fixed free-VRAM gate.

    Topaz owns its VRAM budget through ``vram``. A fixed 12 GiB launch reserve
    rejected otherwise usable 16 GiB cards and did not describe actual model
    demand. The shared guard still catches invalid telemetry, critical margins,
    and sustained low resources while the child is running.
    """
    row = reader.sample()
    log.write(json.dumps({'phase': 'startup', **row}) + '\n')
    log.flush()
    reason = guard.observe(row)
    if reason:
        raise RuntimeError('Topaz resource telemetry failed before launch: ' + reason)
    available_ram = row['ram_available_bytes']
    if available_ram < TOPAZ_STARTUP_FREE_RAM_BYTES:
        required_gib = TOPAZ_STARTUP_FREE_RAM_BYTES / 1024**3
        available_gib = available_ram / 1024**3
        raise RuntimeError(
            f'Topaz needs {required_gib:.1f} GiB available system RAM before launch; '
            f'{available_gib:.1f} GiB is available. No fixed free-VRAM launch reserve is required.')
    return row


class TaskProgress:
    """Translate the owned worker's files into monotonic ComfyUI progress."""
    def __init__(self, job, callback, operation, multiplier=1):
        self.job = Path(job)
        self.callback = callback
        self.operation = operation
        self.multiplier = multiplier
        self.total = None
        self.last = -1

    def emit(self, value):
        value = max(self.last, min(100, int(value)))
        if self.callback is not None and value != self.last:
            self.callback(value)
        self.last = value

    def refresh(self):
        if self.callback is None:
            return
        if (self.job / 'result.json').is_file():
            self.emit(100)
            return
        if (self.job / 'strict_decode.stdout').is_file():
            self.emit(98)
        elif (self.job / 'output_pcm.stderr').is_file():
            self.emit(95)
        elif (self.job / 'output_video_probe.stdout').is_file():
            self.emit(92)
        elif (self.job / 'encoder_probe.stdout').is_file():
            self.emit(4)
        else:
            self.emit(1)
        source_probe = self.job / 'source_video_probe.stdout'
        if self.total is None and source_probe.is_file():
            try:
                self.total = len(json.loads(source_probe.read_text(encoding='utf8')).get('frames', []))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        engine_log = self.job / (self.operation + '.stdout')
        if self.total and engine_log.is_file():
            try:
                frames = re.findall(r'(?m)^frame=(\d+)\s*$', engine_log.read_text(encoding='utf8'))
            except (OSError, UnicodeDecodeError):
                frames = []
            if frames:
                expected = self.total * self.multiplier
                self.emit(5 + 85 * min(int(frames[-1]), expected) / expected)


def run_regular(runtime, source, job, *, model_id, width, height, scale,
                lease_path, device=0, vram=.8, instances=0, parameters=None, interrupt=None, size_mode='scale',
                output_profile='delivery_h264', parameter_audit=None, progress=None):
    """Single owned task, no auto retry/resume, only publish after media audit.

    Fixed scales and explicit same-aspect sizes are checked against actual source
    geometry by the owned worker before enhancement. No parent video decode.
    """
    from .dlss_fi_backend.process import run_isolated, IsolatedTaskError
    from .dlss_fi_backend.resources import SerialProbeLease, NvmlResourceReader, ResourceGuard
    if scale not in (1, 2, 4) or type(scale) is not int or type(device) is not int or not 0 <= device <= 15:
        raise ValueError('Regular Topaz route requires fixed1x/2x/4x and a GPU index in0..15')
    if (width is None) != (height is None):
        raise ValueError('Either provide both target dimensions or infer both from the source')
    if size_mode not in ('scale', 'target_dimensions') or (size_mode == 'target_dimensions' and width is None):
        raise ValueError('Custom size mode requires both target dimensions')
    if output_profile not in REGULAR_OUTPUT_PROFILES:
        raise ValueError('Unknown Topaz output profile')
    regular_filter(runtime, model_id, width if width is not None else 32,
        height if height is not None else 32, device=device, vram=vram,
        instances=instances, parameters=parameters)
    source = Path(source).resolve(strict=True)
    job = Path(job).resolve()
    if job.exists():
        raise ValueError('Use a new task directory; source and previous outputs are never overwritten')
    installation = audit_installation(runtime)
    evidence = (dimension_model_evidence(runtime, model_id) if size_mode == 'target_dimensions'
        else model_evidence(runtime, model_id, scale))
    if parameters and not set(parameters).issubset(installation['tvai_up_options']):
        raise ValueError('Selected parameter is not supported by the installed FFmpeg')
    # Conservative upper bound for an uncompressed lossless master is refined
    # after full source probe in the owned worker, before enhancement starts.
    job.mkdir(parents=True)
    spec = {'install': str(runtime.install), 'definitions': str(runtime.definitions), 'data': str(runtime.data),
        'source': file_identity(source), 'installation': installation, 'model': evidence,
        'parameter_audit': parameter_audit or {'mode': 'api', 'ignored_unsupported_model_controls': []},
        'settings': {'model_id': model_id, 'width': width, 'height': height, 'scale': scale, 'size_mode': size_mode,
            'device': device, 'vram': vram, 'instances': instances, 'parameters': parameters or {},
            'output_profile': output_profile}, 'job': str(job)}
    (job / 'request.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
    receipt = {'status': 'incomplete', 'no_automatic_downloads': True}
    interrupt_error = None
    runtime_disk_floor = 256 * 1024**2 if output_profile in DELIVERY_OUTPUT_PROFILES else 1024**3
    task_progress = TaskProgress(job, progress, 'enhancement')
    task_progress.emit(0)
    try:
        with SerialProbeLease(lease_path), NvmlResourceReader() as reader:
            guard = ResourceGuard()
            with (job / 'resources.jsonl').open('x', encoding='utf8') as log:
                record_topaz_startup_sample(reader, guard, log)
                previous = time.monotonic()
                def check():
                    nonlocal previous, interrupt_error
                    if interrupt:
                        try:
                            interrupt()
                        except BaseException as error:
                            interrupt_error = error
                            raise
                    if time.monotonic() - previous < .25:
                        return
                    previous = time.monotonic()
                    task_progress.refresh()
                    row = reader.sample()
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    reason = guard.observe(row)
                    if reason:
                        raise RuntimeError('Topaz resource guard: ' + reason)
                    if shutil.disk_usage(job).free < runtime_disk_floor:
                        raise RuntimeError('Topaz stopped before filling output disk')
                receipt = run_isolated(Path(__file__).with_name('topaz_worker.py'),
                    [str(job / 'request.json')], timeout=3600, check=check)
    except IsolatedTaskError as error:
        receipt = error.receipt
        # The generic owned-process runner deliberately captures exceptions to
        # ensure job cleanup. Restore Comfy's actual cancellation exception only
        # once it certifies no owned descendants remain; never mask cleanup failure.
        if interrupt_error is not None and receipt.get('active_after_cleanup') == 0:
            raise interrupt_error from error
        raise
    except BaseException as error:
        receipt = {'status': 'controller_failed', 'error': str(error)}
        raise
    finally:
        (job / 'process.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    report = json.loads((job / 'result.json').read_text(encoding='utf8'))
    task_progress.emit(100)
    return validate_publication(job, spec, report), report


def run_interpolation(runtime, source, job, *, model_id, multiplier, lease_path,
                      device=0, vram=.8, instances=0, duplicate_threshold=.01,
                      output_profile='delivery_h264', interrupt=None, progress=None):
    """Run official tvai_fi in an owned process; preserve duration and audio."""
    from .dlss_fi_backend.process import run_isolated, IsolatedTaskError
    from .dlss_fi_backend.resources import SerialProbeLease, NvmlResourceReader, ResourceGuard
    if type(multiplier) is not int or multiplier not in (2, 4):
        raise ValueError('Topaz interpolation multiplier must be2 or4')
    if output_profile not in DELIVERY_OUTPUT_PROFILES:
        raise ValueError('Unknown Topaz interpolation output profile')
    interpolation_filter(runtime, model_id, 24 * multiplier, device=device, vram=vram, instances=instances,
        duplicate_threshold=duplicate_threshold)
    source = Path(source).resolve(strict=True)
    job = Path(job).resolve()
    if job.exists():
        raise ValueError('Use a new task directory; source and previous outputs are never overwritten')
    installation = audit_installation(runtime)
    if not installation.get('tvai_fi_available'):
        raise RuntimeError('This official Topaz installation does not expose the required tvai_fi interface')
    evidence = interpolation_model_evidence(runtime, model_id)
    job.mkdir(parents=True)
    spec = {'install': str(runtime.install), 'definitions': str(runtime.definitions),
        'data': str(runtime.data), 'source': file_identity(source),
        'installation': installation, 'model': evidence,
        'settings': {'model_id': model_id, 'multiplier': multiplier, 'device': device, 'instances': instances,
                     'vram': vram, 'duplicate_threshold': duplicate_threshold,
                     'output_profile': output_profile}, 'job': str(job)}
    (job / 'request.json').write_text(json.dumps(spec, indent=2), encoding='utf8')
    receipt = {'status': 'incomplete', 'no_automatic_downloads': True}
    interrupt_error = None
    task_progress = TaskProgress(job, progress, 'interpolation', multiplier)
    task_progress.emit(0)
    try:
        with SerialProbeLease(lease_path), NvmlResourceReader() as reader:
            guard = ResourceGuard()
            with (job / 'resources.jsonl').open('x', encoding='utf8') as log:
                record_topaz_startup_sample(reader, guard, log)
                previous = time.monotonic()

                def check():
                    nonlocal previous, interrupt_error
                    if interrupt:
                        try:
                            interrupt()
                        except BaseException as error:
                            interrupt_error = error
                            raise
                    if time.monotonic() - previous < .25:
                        return
                    previous = time.monotonic()
                    task_progress.refresh()
                    row = reader.sample()
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    reason = guard.observe(row)
                    if reason:
                        raise RuntimeError('Topaz resource guard: ' + reason)
                    if shutil.disk_usage(job).free < 256 * 1024**2:
                        raise RuntimeError('Topaz stopped before filling output disk')

                receipt = run_isolated(Path(__file__).with_name('topaz_fi_worker.py'),
                    [str(job / 'request.json')], timeout=3600, check=check)
    except IsolatedTaskError as error:
        receipt = error.receipt
        if interrupt_error is not None and receipt.get('active_after_cleanup') == 0:
            raise interrupt_error from error
        raise
    except BaseException as error:
        receipt = {'status': 'controller_failed', 'error': str(error)}
        raise
    finally:
        (job / 'process.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    report = json.loads((job / 'result.json').read_text(encoding='utf8'))
    task_progress.emit(100)
    return validate_publication(job, spec, report), report
