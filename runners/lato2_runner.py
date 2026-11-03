"""LATO.2 low-poly re-meshing of one mesh (run with the LATO.2 environment's Python).

    python lato2_runner.py --repo LATO2_DIR --ckpt CKPT_DIR --input mesh.glb --output out.glb
        [--vert_num 2000] [--seed 42] [--vflow_steps 24] [--tflow_steps 50] [--cfg 3.0] [--no-solidify]
        [--no-surface] [--dump F.npz] [--condition_image F.png]

The input is re-meshed in LATO.2's normalized frame ([-0.5, 0.5] around the bounding box) and the result is moved
back to the input's position and size, so it can replace the source object directly. LATO.2's own faces (every
triangle of the predicted edge graph) stack overlapping faces and leave holes, so the faces are rebuilt from its edge
logits as one edge-manifold, hole-filled surface (lato2_surface.py; --no-surface keeps LATO.2's faces). Progress
lines start with "[lato2 progress]".
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import trimesh

# The defaults of Make Solid in StableGen (https://github.com/sakalond/StableGen), whose exterior fill this is.
CLOSE_HOLES = 3
VIEW_DIRECTIONS = 64


def progress(value, stage):
    print(f"[lato2 progress] {value} {stage}", flush=True)


def load_single_mesh(path):
    loaded = trimesh.load(path, force='scene', process=False)
    meshes = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    if not meshes:
        raise RuntimeError(f"no triangles in {path}")
    mesh = trimesh.util.concatenate(meshes)
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces), process=False)


def patch_surface(module, args, source, center, extent):
    """Keep every vertex pair's edge logit above CANDIDATE_LOGIT while LATO.2 decodes its topology, and rebuild the
    faces from them just before it writes the mesh. With --tflow_samples K the topology flow runs K times and the
    logits are averaged."""
    import torch
    from lato2_surface import CANDIDATE_LOGIT, rebuild

    captured = []
    samples = []
    sample = module.TopoFlowEulerSampler.sample

    def sample_many(self, model, noise, verts, mask, cond=None, cond_mask=None, steps=50):
        first = sample(self, model, noise, verts, mask, cond, cond_mask, steps)
        samples[:] = [first] + [sample(self, model, torch.randn_like(noise), verts, mask, cond, cond_mask, steps)
                                for _ in range(args.tflow_samples - 1)]
        return first

    def decode_with_logits(self, z, verts, verts_mask=None, chunk_size=20000, threshold=0.0):
        # TopologyVAE.decode, returning the edges with a logit above threshold and keeping the near misses.
        latents = [z]
        if len(samples) > 1:
            # main() hands over the first sample divided by the latent scale; the others get the same scale.
            scale = samples[0].float().norm() / z.float().norm()
            latents += [extra.float() / scale for extra in samples[1:]]
        rope = self.connection_rope(verts.long())
        all_feats = []
        for latent in latents:
            feats = self.decoder(z=latent, verts_mask=verts_mask)
            for block in self.connection_transformer_blocks:
                feats = block(feats, verts_mask, rope_phases=rope)
            all_feats.append(feats)
        edges = []
        for i in range(verts_mask.shape[0]):
            items = [feats[i][verts_mask[i]] for feats in all_feats]
            count = int(items[0].shape[0])
            u_idx, v_idx = torch.triu_indices(count, count, offset=1, device=z.device)
            kept_pairs, kept_logits = [], []
            for start in range(0, u_idx.numel(), chunk_size):
                cu, cv = u_idx[start:start + chunk_size], v_idx[start:start + chunk_size]
                logits = sum(self.connection_predictor(item[cu].unsqueeze(0), item[cv].unsqueeze(0)).squeeze(0).float()
                             for item in items) / len(items)
                take = logits > CANDIDATE_LOGIT
                kept_pairs.append(torch.stack([cu[take], cv[take]], dim=-1))
                kept_logits.append(logits[take])
            pairs = torch.cat(kept_pairs).cpu().numpy() if kept_pairs else np.empty((0, 2), dtype=np.int64)
            logits = torch.cat(kept_logits).cpu().numpy() if kept_logits else np.empty(0, dtype=np.float32)
            captured.append((count, pairs, logits))
            edges.append(pairs[logits > threshold])
        return edges

    export_mesh = module.export_mesh

    def export_rebuilt(out_dir, base_name, vert_int, vert_offsets, faces, resolution):
        res = float(resolution)
        positions = vert_int.astype(np.float64) / res - 0.5 + vert_offsets.astype(np.float64) / (res * 2.0)
        match = [item for item in captured if item[0] == len(vert_int)]
        normalized = ((np.asarray(source.vertices) - center) / extent, np.asarray(source.faces))
        rebuilt = None
        if match and args.surface:
            t0 = time.time()
            rebuilt, info = rebuild(positions, match[-1][1], match[-1][2], source=normalized)
            print(f"[lato2] surface rebuilt in {time.time() - t0:.1f}s: {len(faces)} -> {len(rebuilt)} faces {info}",
                  flush=True)
        if args.dump and match:
            np.savez_compressed(args.dump, vertices=positions, pairs=match[-1][1], logits=match[-1][2],
                                lato_faces=np.asarray(faces), faces=rebuilt if rebuilt is not None else np.asarray(faces),
                                source_vertices=normalized[0], source_faces=normalized[1])
        export_mesh(out_dir, base_name, vert_int, vert_offsets,
                    rebuilt if rebuilt is not None and len(rebuilt) else faces, resolution)

    module.TopologyVAE.decode = decode_with_logits
    module.export_mesh = export_rebuilt
    if args.tflow_samples > 1:
        module.TopoFlowEulerSampler.sample = sample_many


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--dino_hub_dir', default=None)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--vert_num', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vflow_steps', type=int, default=24)
    parser.add_argument('--tflow_steps', type=int, default=50)
    parser.add_argument('--cfg', type=float, default=3.0)
    parser.add_argument('--azimuth', type=float, default=45.0)
    parser.add_argument('--elevation', type=float, default=30.0)
    parser.add_argument('--renderer', default='auto', choices=['auto', 'nvdiffrast', 'numpy', 'open3d'],
                        help='conditioning view renderer: auto takes nvdiffrast where it works, else the CPU '
                             '(numpy) one; open3d is LATO.2\'s own')
    parser.add_argument('--solidify', action=argparse.BooleanOptionalAction, default=True,
                        help='drop faces that empty space from outside cannot reach (inner shells) first')
    parser.add_argument('--surface', action=argparse.BooleanOptionalAction, default=True,
                        help='rebuild the faces from the edge logits as one manifold, hole-filled surface')
    parser.add_argument('--tflow_samples', type=int, default=1,
                        help='topology flow samples whose edge logits are averaged')
    parser.add_argument('--dump', default=None, help='save vertices, edge logits and both face sets (.npz)')
    parser.add_argument('--condition_image', default=None,
                        help='save the view LATO.2 conditions on (the grey render fed to DINOv2) as this PNG')
    args = parser.parse_args()
    # The runner changes into the LATO.2 repository below, so relative paths would break.
    for key in ('repo', 'ckpt', 'dino_hub_dir', 'input', 'output', 'dump', 'condition_image'):
        if getattr(args, key):
            setattr(args, key, os.path.abspath(getattr(args, key)))

    progress(2, 'Preparing mesh')
    source = load_single_mesh(args.input)
    if args.solidify:
        # TRELLIS.2 meshes carry a second, inner copy of their surface; LATO.2 would mesh both walls.
        from exterior_fill import exterior_triangles
        source.merge_vertices()
        # glTF is Y-up; a hollow mesh open at the bottom is closed there, and hairline gaps are closed while every
        # face visible from outside stays, as StableGen's Make Solid does by default.
        mask, _info = exterior_triangles(np.asarray(source.vertices), np.asarray(source.faces), seal_bottom=True,
                                         up_axis=1, close_holes=CLOSE_HOLES, view_directions=VIEW_DIRECTIONS)
        if mask.any() and not mask.all():
            print(f"[lato2] removed {int((~mask).sum())} of {len(mask)} interior faces before remeshing", flush=True)
            source = trimesh.Trimesh(vertices=np.asarray(source.vertices), faces=np.asarray(source.faces)[mask],
                                     process=False)
            source.remove_unreferenced_vertices()
    lo, hi = source.bounds
    center = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    work = tempfile.mkdtemp(prefix='lato2_')
    try:
        mesh_dir = os.path.join(work, 'in')
        out_dir = os.path.join(work, 'out')
        os.makedirs(mesh_dir)
        source.export(os.path.join(mesh_dir, 'mesh.obj'))

        # Stand-ins for o_voxel, torch_scatter and flash-attn/xformers where they are not installed (Windows).
        import portable
        portable.install()
        sys.path.insert(0, args.repo)
        os.chdir(args.repo)
        sys.argv = [
            'e2e_inference.py', '--mesh_dir', mesh_dir, '--out_dir', out_dir,
            '--vert_num', str(args.vert_num), '--seed', str(args.seed), '--num_workers', '0',
            '--vflow_steps', str(args.vflow_steps), '--tflow_steps', str(args.tflow_steps),
            '--cfg_strength', str(args.cfg), '--render_azimuth', str(args.azimuth),
            '--render_elevation', str(args.elevation),
            '--vflow_ckpt', os.path.join(args.ckpt, 'vflow.pt'), '--vvae_ckpt', os.path.join(args.ckpt, 'vvae.pt'),
            '--offset_head_ckpt', os.path.join(args.ckpt, 'offset_head.pt'),
            '--tflow_ckpt', os.path.join(args.ckpt, 'tflow.pt'), '--tvae_ckpt', os.path.join(args.ckpt, 'tvae.pt'),
            '--voxel_encoder_ckpt', os.path.join(args.ckpt, 'voxel_encoder.pt'),
            '--dino_hub_dir', args.dino_hub_dir or os.path.join(args.ckpt, 'dinov2'),
        ]
        if args.renderer != 'open3d':
            # Open3D's offscreen renderer returns black images where EGL/Vulkan rendering is unavailable, which
            # would leave LATO.2 without its image condition.
            import dataset.mesh_render as mesh_render
            import white_model_render
            white_model_render.BACKEND = args.renderer
            mesh_render.WhiteModelRenderer = white_model_render.WhiteModelRenderer
        progress(10, 'Generating vertices and topology')
        t0 = time.time()
        import importlib.util
        spec = importlib.util.spec_from_file_location('lato2_e2e', os.path.join(args.repo, 'scripts', 'e2e_inference.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if args.surface or args.dump:
            patch_surface(module, args, source, center, extent)
        module.main()
        result_path = os.path.join(out_dir, 'mesh_pred.obj')
        if not os.path.isfile(result_path):
            raise RuntimeError('LATO.2 produced no mesh (see the log above)')
        progress(90, 'Exporting mesh')
        result = trimesh.load(result_path, force='mesh', process=False)
        vertices = np.asarray(result.vertices, dtype=np.float64) * extent + center
        out = trimesh.Trimesh(vertices=vertices, faces=np.asarray(result.faces), process=False)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        out.export(args.output)
        render = os.path.join(out_dir, 'mesh_render.png')
        if args.condition_image and os.path.isfile(render):
            shutil.copy(render, args.condition_image)
        print(f"[lato2] {len(source.vertices)} -> {len(out.vertices)} vertices, {len(out.faces)} faces "
              f"in {time.time() - t0:.1f}s -> {args.output}", flush=True)
        progress(100, 'Done')
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == '__main__':
    main()
