# Wraptor Visualization Tools

Self-contained visualization for cubed-sphere solvers. Uses solver's own geometry - no external files needed!

## Quick Start

### Run solver and generate movie
```bash
# Diffusion (Lima Flag) - 10 days
python Analysis/visualize_sphere.py --solver diffusion --days 10 --fps 15

# Advection (Cosine Bell) - 12 days 
python Analysis/visualize_sphere.py --solver advection --days 12 --fps 20
```

### High-quality production run
```bash
python Analysis/visualize_sphere.py \
    --solver diffusion \
    --N 120 \
    --days 30 \
    --save-freq 12 \
    --dpi 200 \
    --nlat 360 --nlon 720 \
    --fps 20
```

### Quick preview (lower resolution)
```bash
python Analysis/visualize_sphere.py \
    --solver advection \
    --N 60 \
    --days 6 \
    --save-freq 3 \
    --dpi 100 \
    --nlat 90 --nlon 180 \
    --fps 15
```

## Command-Line Options

### Solver Options
| Option | Description | Default |
|--------|-------------|---------|
| `--solver` | `diffusion` or `advection` | required |
| `--N` | Grid resolution per face | 60 |
| `--days` | Simulation duration | 10 |
| `--save-freq` | Output frequency (hours) | 6 |

### Output Options
| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Output directory | `output` |
| `--fps` | Movie frames per second | 15 |
| `--dpi` | Frame resolution | 150 |

### Visualization Options
| Option | Description | Default |
|--------|-------------|---------|
| `--cmap` | Colormap name | `inferno` |
| `--vmin` | Color scale min | auto |
| `--vmax` | Color scale max | auto |
| `--log-scale` | Logarithmic colors | off |
| `--elev` | Viewing elevation (°) | 30 |
| `--azim` | Viewing azimuth (°) | 45 |
| `--nlat` | Lat interpolation points | 180 |
| `--nlon` | Lon interpolation points | 360 |

## Viewing Angles

### Recommended views:
- **Standard 3/4 view:** `--elev 30 --azim 45` (default)
- **North pole focus:** `--elev 60 --azim 45`
- **Equatorial view:** `--elev 0 --azim 45`
- **Earth tilt:** `--elev 23.5 --azim 0`

## Colormaps

### Temperature (diffusion):
- `inferno` - dark to bright (default)
- `magma` - purple to white
- `plasma` - purple to yellow

### Tracers (advection):
- `viridis` - perceptually uniform
- `cividis` - colorblind-friendly

## Output Structure

```
output/
├── frames/
│   ├── frame_0001.png
│   ├── frame_0002.png
│   └── ...
└── diffusion_movie.mp4  (or advection_movie.mp4)
```

## Requirements

- numpy, scipy, matplotlib
- ffmpeg (for movie generation)

Install ffmpeg:
```bash
brew install ffmpeg   # macOS
sudo apt install ffmpeg  # Linux
```

## Tips

1. **Test first** - Use low resolution (`--N 30 --dpi 100`) for quick testing
2. **Skip bad frames** - First frame (t=0) is automatically skipped
3. **Presentation quality** - Use `--dpi 200 --nlat 360 --nlon 720`
4. **File size** - Higher DPI = larger files. 150 DPI is usually enough.

## Troubleshooting

### "ffmpeg not found"
Install ffmpeg (see Requirements above)

### Movie is choppy
Reduce `--save-freq` for more frames, or increase `--fps`

### Colors washed out
Try `--log-scale` for fields with large dynamic range, or set explicit `--vmin` and `--vmax`

## Presentation Workflow

1. **Preview run (fast check)**  
   ```bash
   python Analysis/visualize_sphere.py \
       --solver diffusion \
       --days 5 \
       --N 60 \
       --save-freq 12 \
       --dpi 100 \
       --nlat 120 --nlon 240 \
       --fps 15
   ```
   - Low-resolution interpolant, shorter movie (~1.5 s), and bypasses heavy I/O.
   - Use this step to verify viewpoint, colormap, and that halo exchange/PLR output looks reasonable.

2. **Production movie (presentation-ready)**  
   ```bash
   python Analysis/visualize_sphere.py \
       --solver diffusion \
       --days 10 \
       --N 120 \
       --save-freq 6 \
       --dpi 200 \
       --nlat 360 --nlon 720 \
       --fps 20
   ```
   - Skips the first frame automatically, so you never start on the “cold” snapshot.
   - The ffmpeg call now forces even resolution (`scale=trunc(iw/2)*2:...`), so H.264 encoding succeeds.
   - Playback file lives in `output/diffusion_movie.mp4` (or `advection_movie.mp4` for PLR).

3. **Presentation checklist**  
   - Confirm the color scale matches what you want (log vs linear, explicit vmin/vmax).  
   - Check runtime stats printed during frame generation: color range, frame count, and estimated duration.  
   - Describe the workflow on your slide: “Run solver → generate frames → ffmpeg movie” and mention the tool uses the solver’s geometry directly.

