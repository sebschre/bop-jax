"""Trajectories for OVITO: atoms with spin and orbital moments as arrows.

``Trajectory`` collects the frames of a molecular dynamics, spin dynamics or
spin-lattice run and hands them to OVITO (https://www.ovito.org) in two ways:

* ``write`` saves an extended XYZ file, which the OVITO program opens
  directly; every per-atom vector appears there as a particle property that
  can be shown as arrows (*Visual elements*, vector display).  This needs
  nothing beyond NumPy.
* ``pipeline`` and ``render`` use the ``ovito`` Python module
  (``pip install ovito``, or the ``viz`` extra of bopjax) to build a data
  pipeline in memory and to render pictures and animated GIFs or movies with
  the arrows already set up.

    from bopjax import ovito as bov

    traj = bov.Trajectory(cell)
    for step in range(steps):
        state = apply(state)
        traj.append(state, time=step * dt_fs, orbital_moment=L)
    traj.write("run.xyz")
    traj.render("run.gif", arrows={"spin_moment": bov.SPIN, "orbital_moment": bov.ORBITAL})

``append`` understands the states of ``bopjax.md.nve``, ``bopjax.spin.llg``
and ``bopjax.spinlattice.nve`` and stores of each what it carries:

    position      -> the atomic positions
    axes, moment  -> "spin_moment" = moment * axes, in Bohr magnetons
    force         -> "force"
    momentum, mass-> "velocity"
    energy        -> the global attribute "energy"

Further keyword arguments add per-atom arrays (N,) or (N, 3) and global
numbers under their own names.

Rendering on a machine without a display uses OVITO's software renderers; from
OVITO 3.13 on these need a Vulkan driver, for which Mesa's software
implementation is enough (``apt install mesa-vulkan-drivers``).
"""

from __future__ import annotations

import glob
import os
import re
import tempfile

import numpy as np

# arrow styles for ``render``: colour (RGB), factor from the property to
# Angstrom, and shaft width in Angstrom
SPIN = {"color": (0.84, 0.15, 0.16), "scale": 1.0, "width": 0.16}
ORBITAL = {"color": (0.12, 0.47, 0.71), "scale": 1.0, "width": 0.16}
FORCE = {"color": (0.17, 0.63, 0.17), "scale": 1.0, "width": 0.12}

_RESERVED = ("species", "pos")


class Trajectory:
    """Frames of a simulation: positions and any per-atom or global quantity.

    cell    : (3, 3) cell vectors as rows, or None for a free cluster
    species : (N,) species index of each atom (default: all 0)
    names   : chemical symbol of each species (default: "X")
    pbc     : periodic directions (default: all if there is a cell)
    """

    def __init__(self, cell=None, species=None, names=("X",), pbc=None):
        self.cell = None if cell is None else np.asarray(cell, dtype=float).reshape(3, 3)
        self.species = None if species is None else np.asarray(species, dtype=int)
        self.names = tuple(names)
        self.pbc = (self.cell is not None,) * 3 if pbc is None else tuple(bool(p) for p in pbc)
        self.frames = []  # (positions, {name: per-atom array}, {name: number})

    def __len__(self):
        return len(self.frames)

    def append(self, state=None, *, positions=None, **quantities):
        """Add a frame from an integrator state and/or explicit arrays."""
        per_atom, attributes = {}, {}
        if state is not None:
            fields = getattr(state, "_fields", ())
            if positions is None and "position" in fields:
                positions = state.position
            if "axes" in fields and "moment" in fields:
                per_atom["spin_moment"] = (np.asarray(state.moment)[:, None]
                                           * np.asarray(state.axes))
            if "force" in fields:
                per_atom["force"] = np.asarray(state.force)
            if "momentum" in fields and "mass" in fields:
                per_atom["velocity"] = np.asarray(state.momentum) / np.asarray(state.mass)
            if "energy" in fields:
                attributes["energy"] = float(state.energy)
        if positions is None:
            raise ValueError("a frame needs positions: pass a state or positions=")
        positions = np.array(positions, dtype=float).reshape(-1, 3)
        n = len(positions)
        if self.species is None:
            self.species = np.zeros(n, dtype=int)
        if len(self.species) != n:
            raise ValueError(f"{n} positions for {len(self.species)} atoms")
        for name, value in quantities.items():
            if name in _RESERVED or not name.isidentifier():
                raise ValueError(f"invalid name of a quantity: {name!r}")
            value = np.array(value, dtype=float)
            if value.ndim == 0:
                attributes[name] = float(value)
            elif value.shape in ((n,), (n, 3)):
                per_atom[name] = value
            else:
                raise ValueError(f"{name!r} has shape {value.shape}; expected a number, "
                                 f"({n},) or ({n}, 3)")
        if self.frames and set(per_atom) != set(self.frames[0][1]):
            raise ValueError("every frame must carry the same per-atom quantities")
        self.frames.append((positions, per_atom, attributes))

    # --- file output -------------------------------------------------------------

    def _symbols(self):
        extra = range(len(self.names), 1 + int(self.species.max()))
        names = self.names + tuple(f"X{k}" for k in extra)
        return [names[k] for k in self.species]

    def write(self, path):
        """Write all frames as extended XYZ, which OVITO (and ASE) read.  The
        per-atom quantities become columns under their names."""
        if not self.frames:
            raise ValueError("the trajectory has no frames")
        symbols = self._symbols()
        with open(path, "w") as out:
            for positions, per_atom, attributes in self.frames:
                columns, layout = [positions], "species:S:1:pos:R:3"
                for name, value in per_atom.items():
                    columns.append(value.reshape(len(positions), -1))
                    layout += f":{name}:R:{columns[-1].shape[1]}"
                header = []
                if self.cell is not None:
                    lattice = " ".join(f"{x:.10g}" for x in self.cell.ravel())
                    header.append(f'Lattice="{lattice}"')
                header.append(f"Properties={layout}")
                header.append('pbc="' + " ".join("T" if p else "F" for p in self.pbc) + '"')
                header += [f"{name}={value:.12g}" for name, value in attributes.items()]
                out.write(f"{len(positions)}\n{' '.join(header)}\n")
                for symbol, row in zip(symbols, np.hstack(columns)):
                    out.write(symbol + "".join(f" {x:.8f}" for x in row) + "\n")
        return path

    @classmethod
    def read(cls, path):
        """Read a file written by ``write`` (not general extended XYZ)."""
        with open(path) as handle:
            lines = handle.read().splitlines()
        traj, at, order = None, 0, []
        while at < len(lines) and lines[at].strip():
            n = int(lines[at])
            header = dict((k, a if a else b) for k, a, b in
                          re.findall(r'(\w+)=(?:"([^"]*)"|(\S+))', lines[at + 1]))
            rows = [line.split() for line in lines[at + 2:at + 2 + n]]
            if traj is None:
                order = sorted(set(r[0] for r in rows), key=[r[0] for r in rows].index)
                cell = (np.array(header["Lattice"].split(), dtype=float).reshape(3, 3)
                        if "Lattice" in header else None)
                traj = cls(cell, [order.index(r[0]) for r in rows], order,
                           [p == "T" for p in header["pbc"].split()])
            values = np.array([r[1:] for r in rows], dtype=float)
            layout = header["Properties"].split(":")[3:]
            per_atom, col = {}, 0
            for name, _, width in zip(layout[0::3], layout[1::3], layout[2::3]):
                block = values[:, col:col + int(width)]
                col += int(width)
                if name != "pos":
                    per_atom[name] = block[:, 0] if int(width) == 1 else block
            attributes = {k: float(v) for k, v in header.items()
                          if k not in ("Lattice", "Properties", "pbc")}
            traj.frames.append((values[:, :3], per_atom, attributes))
            at += 2 + n
        return traj

    # --- OVITO -------------------------------------------------------------------

    def pipeline(self, arrows=None, radius=0.4, color=(0.65, 0.65, 0.68), exaggerate=1.0,
                 replicate=None, show_cell=True, margin=3.0):
        """An ``ovito.pipeline.Pipeline`` that serves the frames.

        arrows     : {name of an (N, 3) quantity: style}; a style is a dict with
                     "color", "scale" (Angstrom per unit of the quantity) and
                     "width", see ``SPIN``.  Default: "spin_moment" if present.
        radius     : radius of the atoms (a number, or one per species)
        color      : colour of the atoms (RGB, or one per species)
        exaggerate : factor on the displacements from the first frame, to show
                     motions that are too small to see
        replicate  : (nx, ny, nz) periodic images to show
        margin     : space around a free cluster in its (hidden) box
        """
        _ovito()
        from ovito.data import ParticleType
        from ovito.modifiers import ReplicateModifier
        from ovito.pipeline import Pipeline, PipelineSourceInterface, PythonSource
        from ovito.vis import VectorVis

        if not self.frames:
            raise ValueError("the trajectory has no frames")
        frames, species = self.frames, self.species
        symbols = sorted(set(zip(species.tolist(), self._symbols())))
        nspecies = 1 + int(species.max())
        radii = np.broadcast_to(np.asarray(radius, dtype=float), (nspecies,))
        colors = np.broadcast_to(np.asarray(color, dtype=float), (nspecies, 3))
        if arrows is None:
            arrows = {"spin_moment": SPIN} if "spin_moment" in frames[0][1] else {}
        visuals = {}
        for name, style in arrows.items():
            if name not in frames[0][1] or frames[0][1][name].ndim != 2:
                raise ValueError(f"no per-atom vector {name!r} in the trajectory")
            style = dict(SPIN, **style)
            visuals[name] = VectorVis(
                enabled=True, color=tuple(style["color"]), scaling=float(style["scale"]),
                width=float(style["width"]), alignment=VectorVis.Alignment.Base,
                shading=VectorVis.Shading.Normal)
        reference = frames[0][0]
        if self.cell is None:
            low = np.min([f[0].min(axis=0) for f in frames], axis=0) - margin
            high = np.max([f[0].max(axis=0) for f in frames], axis=0) + margin
            matrix = np.column_stack([np.diag(high - low), low])
        else:
            matrix = np.column_stack([self.cell.T, np.zeros(3)])
        pbc, visible = self.pbc, show_cell and self.cell is not None

        class Source(PipelineSourceInterface):
            def compute_trajectory_length(self, **kwargs):
                return len(frames)

            def create(self, data, *, frame, **kwargs):
                positions, per_atom, attributes = frames[frame]
                cell = data.create_cell(matrix, pbc=pbc)
                cell.vis.enabled = visible
                cell.vis.rendering_color = (0.45, 0.45, 0.45)
                particles = data.create_particles(count=len(positions))
                particles.create_property(
                    "Position", data=reference + exaggerate * (positions - reference))
                types = particles.create_property("Particle Type", data=species + 1)
                for k, symbol in symbols:
                    kind = next((t for t in types.types if t.id == k + 1), None)
                    if kind is None:
                        kind = ParticleType(id=k + 1)
                        types.types.append(kind)
                    kind = types.make_mutable(kind)
                    kind.name, kind.radius, kind.color = symbol, float(radii[k]), tuple(colors[k])
                for name, value in per_atom.items():
                    if value.ndim == 2:
                        prop = particles.create_property(name, dtype=float, components=3,
                                                         data=value)
                        prop.vis = visuals.get(name)
                    else:
                        particles.create_property(name, dtype=float, data=value)
                data.attributes["frame"] = frame
                for name, value in attributes.items():
                    data.attributes[name] = value

        pipeline = Pipeline(source=PythonSource(delegate=Source()))
        if replicate is not None:
            nx, ny, nz = replicate
            pipeline.modifiers.append(ReplicateModifier(num_x=nx, num_y=ny, num_z=nz))
        return pipeline

    def render(self, path, arrows=None, size=(480, 480), fps=20, every=1,
               camera_dir=(-1.0, 2.0, -0.6), camera_up=(0.0, 0.0, 1.0), perspective=False,
               zoom=1.0, label=None, legend=True, background=(1.0, 1.0, 1.0),
               renderer="tachyon", colors=128, **pipeline_options):
        """Render the trajectory with OVITO.

        path       : ".gif", ".mp4", ".avi" or ".mov" for an animation; an image
                     file (".png", ".jpg") for the first frame only
        arrows     : as in ``pipeline``
        label      : text in the upper left corner, may have several lines;
                     ``{name}`` inserts a global attribute of the frame, e.g.
                     "t = {time:.0f} fs"
        legend     : name each kind of arrow in its colour (a dict
                     {name: text} sets the texts)
        zoom       : > 1 enlarges the picture
        renderer   : "tachyon" (ray tracing with ambient occlusion), "opengl"
                     (much faster), or an OVITO renderer object
        colors     : size of the palette shared by the frames of a GIF; GIFs are
                     assembled with Pillow when it is installed (fewer colours,
                     smaller files)
        Further keyword arguments go to ``pipeline``.
        """
        _ovito()
        from ovito.vis import OpenGLRenderer, TachyonRenderer, Viewport

        pipeline = self.pipeline(arrows, **pipeline_options)
        if arrows is None:
            arrows = {"spin_moment": SPIN} if "spin_moment" in self.frames[0][1] else {}
        if renderer == "tachyon":
            renderer = TachyonRenderer(shadows=False, ambient_occlusion=True,
                                       ambient_occlusion_brightness=0.9)
        elif renderer == "opengl":
            renderer = OpenGLRenderer()
        pipeline.add_to_scene()
        try:
            viewport = Viewport(
                type=Viewport.Type.Perspective if perspective else Viewport.Type.Ortho,
                camera_dir=tuple(camera_dir), camera_up=tuple(camera_up))
            viewport.zoom_all(size=tuple(size))
            if perspective:
                viewport.fov = 2.0 * np.arctan(np.tan(0.5 * viewport.fov) / zoom)
            else:
                viewport.fov = viewport.fov / zoom
            font = max(0.02, 15.0 / size[1])
            if label is not None:
                def text_of(frame):
                    return label.format(frame=frame, **self.frames[frame][2])

                text_of(0)  # fail early on unknown names
                overlay = _frame_label(text_of, font)
                viewport.overlays.append(overlay)
            if legend:
                texts = legend if isinstance(legend, dict) else {}
                entries = [(texts.get(name, name.replace("_", " ")),
                            tuple(dict(SPIN, **style)["color"])) for name, style in arrows.items()]
                viewport.overlays.append(_legend(entries, font))
            options = dict(size=tuple(size), background=tuple(background), renderer=renderer)
            kind = os.path.splitext(str(path))[1].lower()
            if kind == ".gif" and _pillow() is not None:
                # OVITO's GIFs carry a palette per frame; one shared, smaller
                # palette keeps the file smaller and the colours steady
                with tempfile.TemporaryDirectory() as folder:
                    viewport.render_anim(filename=os.path.join(folder, "frame*.png"),
                                         every_nth=every, **options)
                    files = glob.glob(os.path.join(folder, "frame*.png"))
                    files.sort(key=lambda name: int(re.sub(r"\D", "", os.path.basename(name))))
                    _assemble_gif(files, path, fps, colors)
            elif kind in (".gif", ".mp4", ".avi", ".mov"):
                viewport.render_anim(filename=str(path), fps=fps, every_nth=every, **options)
            else:
                viewport.render_image(filename=str(path), frame=0, **options)
        finally:
            pipeline.remove_from_scene()
        return path


def _frame_label(text_of, font):
    """Overlay that draws ``text_of(frame)`` in the upper left corner."""
    from ovito.vis import PythonViewportOverlay, ViewportOverlayInterface

    class Label(ViewportOverlayInterface):
        def render(self, canvas, *, frame, **kwargs):
            for row, line in enumerate(text_of(frame).split("\n")):
                canvas.draw_text(line, pos=(0.02, 0.98 - 1.5 * font * row), font_size=font,
                                 anchor="north west", color=(0.1, 0.1, 0.1))

    return PythonViewportOverlay(delegate=Label())


def _pillow():
    try:
        import PIL.Image
    except ImportError:  # pragma: no cover
        return None
    return PIL.Image


def _assemble_gif(files, path, fps, colors):
    """One palette, taken from up to eight frames, for all frames."""
    image = _pillow()
    frames = [image.open(name).convert("RGB") for name in files]
    sheet = image.new("RGB", (frames[0].width, frames[0].height * min(len(frames), 8)))
    for k, frame in enumerate(frames[::max(1, len(frames) // 8)][:8]):
        sheet.paste(frame, (0, k * frames[0].height))
    palette = sheet.quantize(colors=colors, method=image.Quantize.MEDIANCUT)
    frames = [f.quantize(palette=palette, dither=image.Dither.NONE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps),
                   loop=0, optimize=False, disposal=1)


def _legend(entries, font):
    """Overlay that lists (text, colour) in the lower left corner."""
    from ovito.vis import PythonViewportOverlay, ViewportOverlayInterface

    class Legend(ViewportOverlayInterface):
        def render(self, canvas, **kwargs):
            for row, (text, color) in enumerate(reversed(entries)):
                canvas.draw_text(text, pos=(0.02, 0.02 + 1.5 * font * row), font_size=font,
                                 anchor="south west", color=color)

    return PythonViewportOverlay(delegate=Legend())


def _ovito():
    try:
        import ovito
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "rendering needs the OVITO Python module: pip install ovito "
            '(or pip install "bopjax[viz]"); Trajectory.write needs nothing') from error
    return ovito
