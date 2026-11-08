"""CPU-only plotting contracts; synthetic fixtures never become benchmark outputs."""

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import plots as p


def fixture_run(folder=Path("missing"), phase=Path("calibration_v2"), *, seed=0, config=None):
    job = dict(data="fire", method="kgdm", seed=seed, config=config or {}, steps=4000,
               score_test=False, validation_draws=1)
    done = dict(data="fire", method="kgdm", seed=seed, steps=4000, selected_step=4000,
                selected_validation_mmd2=0.01, test_evaluations=0, test_mmd2=None)
    return p.Run(folder, phase, job, done)


def fixture_metrics(run, values=(0.01,), split="val"):
    return [dict(data=run.key[0], method=run.key[1], seed=run.key[2], step=run.selected_step,
                 split=split, draw=i, model_forward_calls=1, kernel_mmd2_v=v,
                 minimum_matching_distance=v + 10) for i, v in enumerate(values)]


def write_run(run):
    run.folder.mkdir(parents=True)
    (run.folder / "job.json").write_text(json.dumps(run.job), encoding="utf-8")
    (run.folder / "completed.json").write_text(json.dumps(run.done), encoding="utf-8")


class PlotContracts(unittest.TestCase):
    def test_periodic_density_and_integral(self):
        points = np.random.default_rng(13).uniform(-np.pi, np.pi, (500, 7))
        a = p.torus_density(points, (5, 6))[2]
        b = p.torus_density(points + 2 * np.pi, (5, 6))[2]
        np.testing.assert_allclose(a, b, atol=1e-12)
        self.assertAlmostEqual(float(a[1:-1, 1:-1].sum() * (2 * np.pi / p.TORUS_BINS)**2), 1.0)
        np.testing.assert_array_equal(a[0], a[-2])
        np.testing.assert_array_equal(a[-1], a[1])

    def test_sphere_radial_normalization(self):
        points = np.eye(3)
        np.testing.assert_allclose(p.sphere_density(points)[2], p.sphere_density(points * 3)[2])
        with self.assertRaises(ValueError):
            p.sphere_density(np.zeros((1, 3)))

    def test_spherical_density_integrates_to_one(self):
        density = p.sphere_density(np.eye(3))[2]
        edges = np.linspace(-np.pi / 2, np.pi / 2, density.shape[0] + 1)
        areas = np.diff(np.sin(edges))[:, None] * 2 * np.pi / density.shape[1]
        self.assertAlmostEqual(float((density * areas).sum()), 1.0, delta=0.0005)

    def test_rna_marginals_cover_all_seven_angles(self):
        self.assertEqual(len(p.RNA_PAIRS), 6)
        self.assertEqual(set(i for pair in p.RNA_PAIRS for i in pair), set(range(7)))
        self.assertEqual(len(p.DATASETS), 9)

    def test_training_seed_sd_is_not_draw_sd(self):
        rows = []
        for seed, value in enumerate((1., 2., 3.)):
            run = fixture_run(seed=seed)
            rows.append(p.summarize_seed(run, fixture_metrics(run, (value - .1, value, value + .1)), "val"))
        result = p.aggregate_seeds(rows)[0]
        self.assertEqual(result["kernel_mmd2_v_mean"], 2)
        self.assertEqual(result["kernel_mmd2_v_training_seed_sd"], 1)
        self.assertAlmostEqual(rows[0]["kernel_mmd2_v_draw_sd"], .1)
        self.assertIsNone(p.aggregate_seeds(rows[:1])[0]["kernel_mmd2_v_training_seed_sd"])
        self.assertIsNone(p.summarize_seed(run, fixture_metrics(run), "val")["kernel_mmd2_v_draw_sd"])
        self.assertEqual(result["minimum_matching_distance_mean"], 12)

    def test_variants_and_configs_are_not_pooled(self):
        rows = []
        for seed, variant in enumerate(("rmf", "rmf_mt")):
            run = fixture_run(seed=seed, config={"variant": variant})
            rows.append(p.summarize_seed(run, fixture_metrics(run), "val"))
        aggregate = p.aggregate_seeds(rows)
        self.assertEqual(len(aggregate), 2)
        self.assertEqual({r["variant"] for r in aggregate}, {"rmf", "rmf_mt"})
        self.assertTrue(all(r["n_seeds"] == 1 for r in aggregate))

    def test_phases_are_not_pooled_and_missing_seeds_are_explicit(self):
        a = fixture_run(seed=0)
        b = fixture_run(phase=Path("mgm_calibration_v1"), seed=1)
        rows = [p.summarize_seed(r, fixture_metrics(r), "val") for r in (a, b)]
        aggregate = p.aggregate_seeds(rows)
        self.assertEqual(len(aggregate), 2)
        self.assertEqual(aggregate[0]["missing_seeds"], "1,2")

    def test_duplicate_seed_cannot_inflate_replicates(self):
        run = fixture_run()
        row = p.summarize_seed(run, fixture_metrics(run), "val")
        with self.assertRaises(ValueError):
            p.aggregate_seeds([row, row])

    def test_validation_selection_and_final_duplicate_guard(self):
        a, b = fixture_run(Path("a")), fixture_run(Path("b"))
        a.done["selected_validation_mmd2"] = .2
        self.assertEqual(p.choose_runs([a, b], "val")[a.key].folder, Path("b"))
        a.done["selected_validation_mmd2"] = .01
        self.assertEqual(p.choose_runs([b, a], "val")[a.key].folder, Path("a"))
        with self.assertRaises(ValueError):
            p.choose_runs([a, b], "test")

    def test_frozen_final_identity_matches_across_seeds(self):
        a, b = fixture_run(seed=0), fixture_run(seed=1)
        self.assertEqual(a.config_id, b.config_id)
        self.assertEqual(len(p.choose_runs([a, b], "test")), 2)
        b.job["config"] = {"lr": .0005}
        with self.assertRaises(ValueError):
            p.choose_runs([a, b], "test")

    def test_final_resolved_defaults_and_sources_must_match(self):
        for changed in ({"resolved_config": {"sample_ema": False}}, {"method_source_sha256": "changed"}):
            a, b = fixture_run(seed=0), fixture_run(seed=1)
            b.job.update(changed)
            with self.assertRaises(ValueError):
                p.choose_runs([a, b], "test")

    def test_split_guards_precede_file_access(self):
        inputs = p.Inputs("val")
        with patch.object(p.np, "load") as read_arrays, patch.object(Path, "read_bytes") as read_json:
            with self.assertRaises(ValueError):
                inputs.array("missing.npz", "test_target", held_out=True)
            with self.assertRaises(ValueError):
                inputs.array("test_step0_samples.npz", "generated_draw_0")
            with self.assertRaises(ValueError):
                inputs.read("test_step0_metrics.json")
            read_arrays.assert_not_called()
            read_json.assert_not_called()
        self.assertFalse(p.is_final_phase("calibration_v2"))
        self.assertFalse(p.is_final_phase("finalish"))
        self.assertTrue(p.is_final_phase("final_v1"))

    def test_incomplete_or_test_disabled_final_is_rejected_before_read(self):
        for change in ({"test_evaluations": 0}, {"steps": 3999}, {"test_mmd2": None}):
            run = fixture_run(phase=Path("final_v1"))
            run.job["score_test"] = True
            run.done.update(test_evaluations=3, test_mmd2=.01)
            run.done.update(change)
            inputs = p.Inputs("test")
            with patch.object(inputs, "read") as read:
                with self.assertRaises(ValueError):
                    p.metric_rows(run, "test", inputs)
                read.assert_not_called()
        run.job["score_test"] = False
        with self.assertRaises(ValueError):
            p.metric_rows(run, "test", inputs)

    def test_only_selected_test_checkpoint_can_be_read(self):
        run = fixture_run(phase=Path("final_v1"))
        run.job["score_test"] = True
        run.done.update(test_evaluations=3, test_mmd2=.01)
        inputs = p.Inputs("test")
        with patch.object(inputs, "read") as read:
            with self.assertRaises(ValueError):
                p.metric_rows(run, "test", inputs, 1000)
            read.assert_not_called()

    def test_discovery_ignores_unfinished_jobs_and_root_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase = Path(tmp) / "calibration_v2"
            run = fixture_run(phase / "complete", phase)
            write_run(run)
            (phase / "completed.json").write_text("{}", encoding="utf-8")
            unfinished = phase / "unfinished"
            unfinished.mkdir()
            (unfinished / "job.json").write_text("invalid JSON; must not be opened", encoding="utf-8")
            selected, candidates = p.discover([phase], "val", ["fire"], p.Inputs("val"))
            self.assertEqual(len(selected), 1)
            self.assertEqual(len(candidates), 1)

    def test_nonfinal_test_enabled_job_is_ignored_before_completion_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase = Path(tmp) / "calibration_v2"
            run = fixture_run(phase / "test_enabled", phase)
            run.job["score_test"] = True
            write_run(run)
            (run.folder / "completed.json").write_text("invalid JSON", encoding="utf-8")
            self.assertFalse(p.discover([phase], "val", ["fire"], p.Inputs("val"))[0])
            with self.assertRaises(ValueError):
                p.discover([phase], "test", ["fire"], p.Inputs("test"))

    def test_completion_budget_and_identity_must_match(self):
        for change in ({"steps": 3999}, {"method": "mgm"}, {"selected_step": 4001}):
            with tempfile.TemporaryDirectory() as tmp:
                phase = Path(tmp) / "calibration_v2"
                run = fixture_run(phase / "complete", phase)
                run.done.update(change)
                write_run(run)
                with self.assertRaises(ValueError):
                    p.discover([phase], "val", ["fire"], p.Inputs("val"))

    def test_metric_identity_draws_nfe_and_score_are_checked(self):
        run = fixture_run()
        inputs = p.Inputs("val")
        with patch.object(inputs, "read", return_value=fixture_metrics(run)):
            self.assertEqual(len(p.metric_rows(run, "val", inputs)), 1)
        for change in ({"seed": 2}, {"draw": 1}, {"model_forward_calls": 2},
                       {"kernel_mmd2_v": .3}, {"variant": "wrong"}, {"recipe_id": "wrong"}):
            rows = fixture_metrics(run)
            rows[0].update(change)
            with patch.object(inputs, "read", return_value=rows):
                with self.assertRaises(ValueError):
                    p.metric_rows(run, "val", inputs)

    def test_require_complete_fails_before_heldout_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            phase = base / "calibration_v2"
            write_run(fixture_run(phase / "complete", phase))
            with patch.object(p, "BASE", base), patch.object(p.Inputs, "array") as arrays:
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    p.main(["--phase", str(phase), "--datasets", "fire", "--split", "val", "--require-complete"])
                arrays.assert_not_called()

    def test_tables_expose_identity_uncertainties_and_missing_methods(self):
        run = fixture_run(config={"variant": "rmf_mt"})
        row = p.summarize_seed(run, fixture_metrics(run), "val")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            p.tables(out, [row], p.aggregate_seeds([row]), ["fire"], "val")
            summary = (out / "seed_mean_metrics.md").read_text(encoding="utf-8")
            detail = (out / "tables/fire_metrics.md").read_text(encoding="utf-8")
            for label in ("rmf_mt", run.config_id[:12], "Training-seed SD", "Missing seeds", "MeanFlow"):
                self.assertIn(label, summary)
            self.assertIn("MinimumMatchingDistance", detail)
            self.assertIn("draw SD", detail)
            self.assertIn("Biased kernel MMD2_V", detail)

    def test_common_scale_and_large_png_contract(self):
        fig = p.plt.figure(figsize=(8, 4))
        try:
            with self.assertRaises(ValueError):
                p.save_figure(fig, Path("never_written"), 240, [])
        finally:
            p.plt.close(fig)
        fig, axes = p.plt.subplots(1, 2, figsize=(20, 4))
        try:
            density = p.torus_density(np.array([[0., 0.], [1., 1.]]))
            vmax = float(density[2].max())
            a = p.render_panel(axes[0], density, vmax, False, (0, 1), [])
            b = p.render_panel(axes[1], density, vmax, False, (0, 1), [])
            self.assertEqual((a.norm.vmin, a.norm.vmax), (b.norm.vmin, b.norm.vmax))
            self.assertEqual(a.norm.vmin, vmax * 1e-3)
            self.assertFalse(a.get_rasterized())
        finally:
            p.plt.close(fig)

    def test_training_curve_excludes_critic_warmup(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run = fixture_run(out / "run")
            run.folder.mkdir()
            (run.folder / "curve.json").write_text(json.dumps([
                dict(step=0, loss=9, generator_updated=0), dict(step=100, loss=2, generator_updated=1),
            ]), encoding="utf-8")
            with patch.object(p, "save_figure", side_effect=lambda fig, *a: p.plt.close(fig)):
                p.curves(out, {run.key: run}, ["fire"], p.Inputs("val"), 240, [])
            with (out / "training_loss_curves.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["step"], "100")

    def test_individual_rna_exports_reuse_row_scales_and_cover_every_pair(self):
        points = np.random.default_rng(4).normal(size=(20, 7))
        panels = {"rna": {None: points, "kgdm": points}}
        run = fixture_run()
        run.job["data"] = "rna"
        densities = {("rna", pair): {None: p.torus_density(points, pair), "kgdm": p.torus_density(points, pair)}
                     for pair in p.RNA_PAIRS}
        saved = []

        def inspect(fig, path, dpi, files, *, min_width):
            self.assertGreaterEqual(round(fig.get_figwidth() * dpi), 2000)
            self.assertEqual(min_width, 2000)
            # Check captions and headings in the actual renderer, not a resized image.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for text in fig.texts:
                bounds = text.get_window_extent(renderer)
                self.assertGreaterEqual(bounds.x0, 0)
                self.assertGreaterEqual(bounds.y0, 0)
                self.assertLessEqual(bounds.x1, fig.bbox.x1)
                self.assertLessEqual(bounds.y1, fig.bbox.y1)
            saved.append(path.name)
            p.plt.close(fig)

        with tempfile.TemporaryDirectory() as tmp, patch.object(p, "save_figure", side_effect=inspect):
            records = p.individual_panels(list(densities), panels, densities, {run.key: run}, "val", [], Path(tmp), 200, [])
        self.assertEqual(len(records), 12)
        self.assertIn("rna_pair6_7_target", saved)
        self.assertIn("rna_pair6_7_kgdm", saved)
        for row in records:
            vmax = max(float(d[2].max()) for d in densities[("rna", tuple(row["pair"]))].values())
            self.assertEqual(row["vmax"], vmax)
            self.assertEqual(row["vmin"], vmax * 1e-3)

    def test_comparison_headings_and_footer_do_not_overlap_panel_labels(self):
        points = np.array([[0., 0.], [1., 1.]])
        density = p.torus_density(points)
        panels = {"general": {method: points for method in (None, *p.METHODS)}}
        densities = {("general", (0, 1)): {method: density for method in panels["general"]}}
        selected = {}
        for method in p.METHODS:
            run = fixture_run()
            run.job.update(data="general", method=method)
            selected[run.key] = run

        def inspect(fig, *args):
            try:
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                header_bottom = min(t.get_window_extent(renderer).y0 for t in fig.texts[:2])
                footer_top = fig.texts[-1].get_window_extent(renderer).y1
                for ax in fig.axes[:5]:
                    self.assertLess(ax.title.get_window_extent(renderer).y1, header_bottom - 4)
                    self.assertGreater(ax.xaxis.label.get_window_extent(renderer).y0, footer_top + 4)
            finally:
                p.plt.close(fig)

        with patch.object(p, "save_figure", side_effect=inspect):
            p.comparison_figure(list(densities), panels, densities, selected, "val", [], Path("unused"), 240, [], {})


if __name__ == "__main__":
    unittest.main()
