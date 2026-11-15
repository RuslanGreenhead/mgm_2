# Карта экспериментов и порядок воспроизведения

Все пути относительно корня архива. Описания ниже отделяют фактически
полученные результаты от существующих, но не завершенных launch plans.

## A. Канонический результат для текущей статьи

| Семейство | Главные исходники | Конфигурация и смысл |
| --- | --- | --- |
| Контролируемый real-data блок | `paper/experiments/controlled_suite.py`, `independent_suite.py`, `mmd_control.py` | S2: Volcano/Earthquake/Flood/Fire; T2: General/Glycine/Proline/Pre-Pro; T7: RNA. Seeds 0/1/2, batch256, width128. GMM MGM 2k G steps, RFM 2k/10k field steps, отдельные MLP objective controls |
| Коррекция RNG и sphere metrics | `independent_suite.py`, `correct_sphere_metrics.py`, `verify_independent.py` | 54 GMM/RFM reruns + 18 неизмененных MLP jobs = 72. Финальная authority: `outputs/paper_icml2026/controlled_independent` |
| Stabilization | `paper/experiments/stabilization.py`, `stabilization_refinement.py`, `stabilization_long.py` | General, seed0, только train/validation; periodic features, warmup, 5/10 C steps, smaller G LR; fixed-final probes |
| Randomized torus | `paper/noisy_torus_study.py`, `verify_noisy_torus.py` | General, три amplitudes 0/0.3/0.7, настоящая derivative noisy curve; exploratory, не дополнительный test benchmark |
| Современные методы | `paper/benchmarks2026/` | Отдельный unfitted uniform-base generator; 50k G/field steps, seed0/1/2, validation-selected checkpoint |
| SE(3) | `paper/benchmarks2026/grasp/` | 2 known objects, 3 methods x 3 seeds, fixed-final10k; RFM50/100 inference steps |

## B. Контролируемый блок: почему несколько каталогов

`controlled/` является ранней версией с обнаруженным вмешательством timing
helper в глобальный CUDA RNG. Она оставлена как история, но не заменяет
исправленный primary result. `controlled_independent_attempt1/` также не
является authority: промежуточный wrapper менял constant scaling torus loss.

Финальная `controlled_independent/` сохраняет старые objectives и гиперпараметры,
но разделяет initialization/target/latent/path/flow RNG streams. Evaluation
проверяется на отсутствие изменения train RNG. Дополнительная sphere-v2
коррекция меняет только NN/coverage/reference calculation из сохраненных arrays,
не обучение и не MMD.

Чистое воспроизведение требует полных datasets и нового рабочего output-root.
Архивные completion metadata нельзя выдавать за готовые веса. Общий порядок:

1. Подготовить исходные real-data файлы и фиксированные splits.
2. В чистом output-root выполнить исходную controlled training sequence из
   `paper/experiments/README.md` и `mmd_control.py`, чтобы получить в том числе
   18 MLP objective-control jobs, копируемых финальным протоколом.
3. Создать новую `controlled_independent` snapshot и выполнить worker именно
   из нее. В worker нужны также исходные 18 MLP files, не только их JSON.
4. Выполнить sphere-v2 correction и независимую проверку.
5. Только после `FINAL_TABLES_READY.json` пересобирать финальные таблицы.

Существенные команды из исходного протокола, после подготовки чистых outputs:

```powershell
python -B paper/experiments/controlled_suite.py train --datasets fire general rna volcano earthquake flood glycine proline prepro --mgm-steps 2000 --rfm-steps 10000 --max-seconds 1000000
python -B paper/experiments/mmd_control.py
python -B paper/experiments/independent_suite.py --prepare
python -B outputs/paper_icml2026/controlled_independent/source_snapshot/paper/experiments/independent_suite.py --worker --root "$PWD"
python -B paper/experiments/correct_sphere_metrics.py
python -B paper/experiments/verify_independent.py
```

Первые обучающие команды используют GPU. Нельзя запускать их автоматически
при проверке архива. Не вызывайте `--prepare` поверх включенной исторической
metadata directory; исходный код специально отказывается ее перезаписывать.

Эти команды не являются новым единым «reproduce all» launcher. Полные preconditions,
верификация copied files, диагностики и записи исходного протокола находятся
в `paper/experiments/README.md`. Полный цикл заново при упаковке не выполнялся.

## C. Современный one-step блок

Четыре метода имеют общий интерфейс `create_method`, `update`, `sample`,
`checkpoint`, `load_checkpoint`. MGM/KGDM/RW-Flow-C используют общий point G;
RMF-MT использует source-derived native time-dependent architecture.

| Метод | Реализация | Что нельзя утверждать |
| --- | --- | --- |
| MGM | hidden-flip minimax, true path velocity, full path differentiation | convergence finite critic не следует из population theorem |
| RW-Flow-C | chordal cost, independent generated self bank, finite Sinkhorn, detached drift target | это equation-checked reimplementation, не проверенный запуск авторского кода |
| KGDM | pinned unchanged author spherical field; wrapped-log torus extension | torus adaptation не является авторским опубликованным experiment |
| RMF-MT | Zhong et al., stopped JVP target, native TMLP, PCGrad | это не другой Riemannian MeanFlow Woo et al. и не full-size author reproduction |

Frozen recipe: `outputs/comparison2026/selected_config_v1.json`. Общие width128,
depth3, batch256, LR2e-4. MGM: sphere/T2 critic width256, 10 C/G, LR_C4e-4;
T7 critic128, 5 C/G, LR_C2e-4; warmup500. T2 выбрал path noise0.7 с полной
velocity. RW: epsilon0.3, flow step0.2; Sinkhorn L128/256/1024 для S2/T2/T7.
KGDM: Gaussian field/step1. RMF: `rmf_mt`. Детали в методических `*NOTES.md`.

Новый независимый прогон без подмены исторического plan:

```powershell
python tools/new_comparison_plan.py --phase final_reproduction_v1
# После наличия исходных данных/кэшей, только намеренно:
python -u -m paper.benchmarks2026.run_jobs --plan outputs/comparison2026/plans/final_reproduction_v1.json
```

Helper не запускает GPU. Очередь проверяет frozen sources, сохраняет optimizer
и RNG states и умеет продолжать реальные checkpoints, но **в архиве этих
checkpoints нет**. Полный запуск содержит 108 jobs; 50k G steps плюс critic work
не являются небольшим smoke test.

Только после всех завершенных jobs:

```powershell
python -m paper.benchmarks2026.profile_all --phase final_reproduction_v1
python -m paper.benchmarks2026.audit --phase final_reproduction_v1
python -m paper.benchmarks2026.report --phase final_reproduction_v1
python -m paper.benchmarks2026.plots --phase outputs/comparison2026/final_reproduction_v1 --split test --require-complete
```

Эта последовательность требует вновь полученных weights, data и samples.
На текущем неполном архиве запускать ее как готовую проверку нельзя.

**Известная переносимость:** существующий полный `audit.py` обращается к
source-hash keys с `/`, а некоторые Windows protocols сохраняли `\`.
Перед будущим полным Windows audit потребуется нормализовать lookup keys
(например, через `PureWindowsPath(...).as_posix()`) в live analysis code.
Не изменяйте frozen snapshots и не переписывайте прежние hashes. Частичный
`partial_report.py` уже учитывает этот случай. Это известная оговорка исходного
analysis helper, не признак завершения полного audit текущего benchmark.

Текущий CPU-only пересчет агрегатов `tools/summarize_saved_metrics.py --check`
не требует полного audit, data arrays, inference или CUDA.

## D. SE(3)

Mug: 2000 source poses, 1290 positive, split1032/129/129.
Table: 2000 source poses, 1559 positive, split1247/155/157.
Split seed2026, object geometry известна во всех splits. Normalization взята
из mesh bounds, не из held-out pose statistics. NFE1 для MGM/RMF-MT, NFE50/100
для одного и того же RFM field. MGM имеет дополнительно 50100 critic updates
на training seed. Все primary checkpoints фиксированы на10k.

Pose score: unbiased average четырех product kernels, SO(3) chordal plus
normalized translation distance; bandwidths0.5/1/2/4. Внутри seed оба объекта
имеют одинаковый вес. Distribution evaluation:512 generated poses/object.
Static geometry:100 первых unfiltered draws/object. Object-any@20 является
отдельной грубой proxy-метрикой, а не успехом физического выполнения.

В `source_mt.py` реально исполняются выбранные AST определения pinned author
loss/PCGrad, а не произвольный новый flow-matching loss. При этом PointNet,
шестивекторное представление, RoMa/path/time и optimizer settings адаптированы.
Это не native SDF experiment авторов.

Для нового пилота сначала восстановить assets, создать новый manifest и plan;
не переиспользовать plan с чужими absolute paths и source/data hashes.
Подробный CLI находится в `paper/benchmarks2026/grasp/README.md`. Физическая
метрика успеха остается `null`, даже если исторические targets были FleX-positive.

## E. Исторический геометрический код

Рабочая директория: `external/riemannian-fm/`.

| Скрипт | Назначение |
| --- | --- |
| `rmgm_train.py` | toy sphere/torus/SPD/hyperbolic |
| `rfm_baseline_train.py` | малый сравнительный RFM, регулируемый inference NFE |
| `rmgm_train_real.py`, `rmgm_real_sphere_train.py` | реальные Earth/angles, empirical samplers |
| `rmgm_run_real_earth.py`, `rmgm_run_real_torus.py` | семейства запусков и summaries |
| `rmgm_prepare_torus_data.py` | исходные molecular downloads/preprocessing |
| `rmgm_prepare_mesh_data.py` | eigenfunction targets, Bunny/Spot/maze |
| `rmgm_run_mesh_bunny_spot.py` | шесть surface targets и генераторные варианты |
| `rmgm_mesh_sphere_map_notes.py` | radial sphere-map notes/smoke, не доказательство биекции |
| `rmgm_run_extra_geometries.py` | dry-run планы appendix и high-D tori; запуск только с `--run` |
| `rmgm_official_full_queue.py` | оригинальные RFM iteration budgets из author configs |
| `rmgm_start_official_queue.py` | detached launcher: не использовать как проверочную команду |
| `rmgm_make_highres_comparisons.py`, `rmgm_mesh_compare.py` | plots/metrics по сохраненным артефактам |

Оригинальные full-budget конфиги сохранены в `configs/experiment/*.yaml`.
`rmgm_official_full_queue.py --dry-run` показывает план, но может записывать
служебную metadata. Наличие этого кода не доказывает, что миллион updates
реально завершен. Некоторые launchers содержат `.conda/rmgm/python.exe` или
старый conda path в defaults; перед запуском переопределить CLI paths.

## F. Результаты, которые нельзя смешивать

- Primary independent controls и ранние contaminated controlled runs.
- Test results, validation-only refinement и train memorization probes.
- Три Monte Carlo draws и три training seeds.
- Многомерный RNA score и картинка двух marginal angles.
- Embedded kernel MMD и nearest-neighbor minimum matching distance.
- Pose distribution, rotation validity, static proxy и physical grasp success.
- Completed 50k training budget и более ранний validation-selected checkpoint.
- Финальные 18 современных jobs и все108 запланированных.
- Small-network adaptations и исходные большие модели/авторские NLL-таблицы.

Полные исходные README и frozen protocols сохраняют детали каждой исторической
ветки. В сомнительном случае ориентируйтесь на manifest нужного запуска,
фактический step/count и код, а не на название каталога или ранний optimistic report.
