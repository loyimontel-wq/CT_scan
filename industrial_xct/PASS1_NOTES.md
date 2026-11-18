# Pass 1 notes: foundation modules

Run tests from the folder that *contains* `industrial_xct/`:
`python -m pytest industrial_xct/tests -v` (34 synthetic tests; they check the code, not your scanner data).
Inspect real data (read-only): `python -m industrial_xct.examples.inspect_dataset --xtekct X.xtekct --angles _ctdata.txt [--projections DIR]`

## 1. Technical assessment of the architecture

Sound as specified. Risks and the changes I made for them:

| Risk | Handling |
|---|---|
| Positional pairing of TIFFs and angles | Matching is by identifier only (`match_projections`); unmatched, duplicate and surplus ids are reported; `id_offset` is a hint, never auto-applied. |
| 3000 vs 3001 mismatch | Reported, nothing deleted. By default `require_complete_match=true` makes the run stop until you review it. |
| CoR applied twice (metadata + image estimate) | `geometry.cor_source` is a single enum; `artifacts.cor.enabled` must agree with it (validated). |
| Zero values in metadata look "measured" | Zero `DetectorOffset*` / `CentreOfRotation*` are flagged: measured zero and "never calibrated" are indistinguishable. |
| `WhiteLevel` taken as I0 | Stored as `white_level_raw`, explicitly unverified. `scalar_i0` mode needs an explicit `i0_value`. |
| Nikon TIFFs may already be flat-corrected | Hence three normalisation modes. Which applies to your data is decided by the intensity inspection (section 5), not assumed. |
| ASTRA API details (filters, cutoff, conventions) | Not verifiable here (ASTRA is not installed in this environment). `filter_cutoff` is rejected by validation until Pass 2 checks it. |
| `.vgi` layout | Written from memory of the common layout, flagged unverified; the raw `.vol` and the JSON sidecar are exact. |
| Short-scan vs full-scan | `classify_angular_coverage` decides before any algorithm is chosen. |

Design change: I added `Provenance`/`ParameterValue`/`Issue` to `config.py` (shared vocabulary) and a `ProjectionSource` interface (TIFF folder and DAT behave identically). Both exist so the Pass 2 report can state where every number came from.

## 2. Findings on your uploaded files (facts from the files only)

* Angle table: 3001 entries, ids 1..3001, header and `.xtekct` both say 3000. Reproduced and reported; no entry removed. Whether entry 3001 has a TIFF cannot be known until I see the folder.
* Steps: median 0.1200 deg (min 0.103, max 0.136): the acquisition is **not perfectly uniform**. This is why per-projection angles, not `InitialAngle + k*AngularStep`, are used.
* Coverage: span 359.644 deg, closure gap 0.236 deg (1.97 steps). Classified `full_scan` under a 1 deg tolerance. That is a tolerance judgement.
* `InitialAngle=12.33` vs first table angle 0.116: a 12.2 deg disagreement, meaning unverified. It only rotates the volume, but matters when comparing orientation with CT Pro 3D.
* Geometry self-consistency: pitch x SOD / SDD = 0.0410002 mm vs metadata VoxelSize 0.0410002 mm (relative difference 2.6e-8). Consistent; this does not prove the distances are correct.
* `DetectorOffsetX/Y`, `CentreOfRotationTop/Bottom` are all 0.0: possibly uncalibrated. Image-based CoR (Pass 2) is mandatory, not optional.
* Fan half-angle 10.12 deg, so a short scan would need at least 200.2 deg; irrelevant here since the scan is full.
* Sampling ratio 0.93 (3000 / ((pi/2) x 2048)): mild angular under-sampling if the object fills the detector width.
* Magnification 4.878; 2048 px acquired region equals the detector size (no ROI).

## 3. Module responsibilities and public interfaces

**config.py**: settings, validation, vocabulary.
`load_config(path, overrides) -> PipelineConfig`, `validate_config(cfg) -> [Issue]`, `validate_or_raise`, `config_to_dict`, `dump_yaml`; `Provenance`, `ParameterValue`, `Issue`.
Unknown YAML keys and wrong types are errors. Defaults enable no correction.

**io_xtek.py**: files in, volumes out.
* Metadata: `parse_xtekct -> XtekMetadata` (all fields preserved, `unconsumed_keys()`), `build_scanner_geometry -> ScannerGeometry` (`.value()`, `.require()`), `validate_geometry`, `expected_image_shape`.
* Angles: `read_angle_file -> AngleTable`, `validate_angle_table -> AngleReport` (has `.unwrapped_deg`), `classify_angular_coverage -> CoverageReport`, `compare_metadata_to_angles`.
* Images: `discover_tiff_projections`, `match_projections`, `require_match`, `build_source(cfg.input, table) -> (source, MatchResult)`; sources expose `.ids`, `.image_shape`, `.dtype`, `.read(i)`, `.read_batch(a,b)`, `.iter_batches(n)`, `.verify_headers()`; `open_dat_stack`; `load_reference_image`.
* Intensity: `inspect_intensity`, `inspect_projection_intensity`.
* Export: `choose_window`, `to_uint16`, `write_tiff_float32`, `write_tiff_uint16`, `write_vol_vgi`.

**preprocessing.py**: counts to line integrals.
`load_calibration`, `normalise_batch`, `verify_air_region`, `detect_bad_pixels`, `replace_masked_pixels`, `beer_lambert`, `preprocess_batch`, `preprocess_stack` (RAM or float32 memmap), `estimate_memory`.

**Interfaces Pass 2 will use for CoR:** `AngleReport.unwrapped_deg` and `MatchedProjection.angle_deg` (true per-projection angles), source `.read()` / the preprocessed stack (pair candidates at ~180 deg apart), `ScannerGeometry` pixel pitch, `cor_offset_px` (single-source state), and `detector_roll/tilt`, `rotation_axis_tilt` kept separate from the horizontal CoR.

## 4. The maths you should be able to derive

1. **Voxel size at the axis** = pitch x SOD / SDD. A ray bundle diverges from the source; an object at the axis is magnified by M = SDD/SOD on the detector.
2. **Transmission** T = (I - D) / (F - D). D removes the detector's electronic offset, F - D is the per-pixel gain x beam profile; dividing removes pixel-to-pixel response and beam non-uniformity.
3. **Line integral** P = -ln T: Beer-Lambert I = I0 exp(-integral of mu dl).
4. **Floor** on T: -ln(0) = infinity. The floor count is reported, because every clamped pixel is replaced information.
5. **Air normalisation** T' = T / T_air: valid only if the region is truly unattenuated; verified per projection, otherwise it refuses.
6. **Unwrapping** with period 360 deg: a jump of more than 180 deg is read as a wrap; coverage = span + one step.
7. **uint16 export**: code = round(clip((x - vmin)/(vmax - vmin), 0, 1) x 65535); recover x = vmin + code(vmax - vmin)/65535; maximum quantisation error is half a level, (vmax - vmin)/131070.

## 5. 16-bit handling (what Pass 1 establishes, and what it does not)

* TIFFs are read unchanged (native dtype, no rescale). Conversion to float32 happens only inside preprocessing; uint16 to float32 is exact.
* `inspect_intensity` reports dtype, min/max, percentiles, bits needed for the maximum, the gcd of occurring levels (detects stretched data), saturation fraction, clipping plateau, and how the data sit relative to `WhiteLevel`.
* Raw counts, transmission, line integrals and reconstructed attenuation are different quantities with different dtype/units; each function documents which it takes and returns.
* Reconstruction values stay float32. 16-bit export always uses an explicit window and writes the scaling to a JSON sidecar and the TIFF description. float32 TIFF is the quantitative alternative.
* **Not yet established:** whether your TIFFs genuinely use 16 bits. This needs the real projection folder (run the inspect script with `--projections`).

## 6. Assumptions and limitations

* `.ang` layout is not verified against a real file; headerless files are parsed as id, angle, extras and anything else is reported as invalid rows.
* Dark/flat are assumed to be single images or folders to average. Nikon scans may have no separate dark/flat at all.
* Bad-pixel detection is a heuristic; check `cal.bad_pixel_mask` visually before enabling.
* The HNW DAT layout is untested against a real file (the NHW path and size validation are tested).
* Nothing here has been validated on real projections yet.

## 7. Practical checks before replying "continue"

1. Run the inspect script with `--projections`: confirm ids 1..3000 or 1..3001, shape 2048x2048, dtype, and read the intensity verdict.
2. Decide which normalisation mode fits: compare the open-beam level in a projection with `WhiteLevel`, and check whether any flat/dark files exist.
3. Open one preprocessed projection and check that air is near 0 in line-integral units and the sample is positive.
4. Export a tiny test volume with `write_vol_vgi` and try importing it in VGSTUDIO MAX (this is the check I cannot do).
