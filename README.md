# MGM for General Geometries: статья и экспериментальный код

For the maintained development layout and the location of each method, see
[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md). The archive description below
documents the original 2026 research bundle.

Архив подготовлен 3 ноября 2026 года. Последняя включенная статья и ее
экспериментальный срез датированы **26 октября 2026 года**. Упаковка не является
новым обучением и не добавляет результатов к этому срезу.

Это исследовательский пакет, а не набор готовых обученных моделей. Здесь есть
исходники архитектур, loss, обучения, оценки, тестов, визуализации и подготовки
статьи. **Весов, чекпойнтов, optimizer/RNG state, установленных окружений и
полных наборов данных нет.**

## 1. С чего начать

1. Открыть текущую статью: [PDF, 36 страниц](output/pdf/Midpoint_Generative_Models_for_General_Geometries_2026-10-26.pdf).
2. Посмотреть [карту экспериментов](docs/EXPERIMENTS.md) и раздел 4 ниже.
3. Проверить целостность: `python tools/verify_bundle.py`.
4. Пересчитать опубликованные агрегаты без GPU и весов:
   `python tools/summarize_saved_metrics.py --check`.
5. Для запуска кода подготовить зависимости и данные по разделам 7-10.

Распаковка, проверка архива и пересчет сохраненных JSON **не запускают обучение**.
Обучающие команды приведены отдельно и выполняются только вручную.

В ZIP используется короткая корневая папка `MGM`. На Windows распаковывайте
архив в достаточно короткий путь: длинные вложенные пути frozen snapshots
могут упираться в ограничение legacy MAX_PATH. Переименовывать внутренние
каталоги исходников и source snapshots ради сокращения пути не нужно.

## 2. Что находится внутри

```text
README.md                         это руководство
MANIFEST.json                     состав, размеры, SHA256 и происхождение файлов
EXCLUSIONS.json                   что исключено и по каким правилам
docs/                             протоколы, данные, ограничения и лицензии
environment/                      наблюдавшиеся версии и базовые зависимости
tools/                            проверка архива и вспомогательные команды
paper/main.tex                    текущая статья
paper/sections/                   разделы статьи, включая новый протокол
paper/generated/                  таблицы TeX из результатов
paper/figures/                    PDF/PNG иллюстрации
paper/experiments/                контролируемый экспериментальный блок
paper/benchmarks2026/             MGM и современные одношаговые бейслайны
paper/benchmarks2026/grasp/       SE(3)-пилот
external/riemannian-fm/           RFM с историческими MGM-расширениями
external/riemannian-fm-official-clean/  отдельная авторская версия RFM
external/kgdm-author/             сохраненный авторский код KGDM
external/riemannian-meanflow-author/   сохраненный авторский код RMF
external/acronym-data/public/     код ACRONYM, без H5/мешей/готовых пакетов
outputs/                         облегченные записи результатов и source snapshots
output/pdf/                      текущий и предыдущий PDF статьи
article_history/                 исходники предыдущей редакции, если доступны
```

Пути сохранены относительно корня проекта, чтобы локальные импорты, конфиги и
структура выходных каталогов оставались понятными. Одинаковые файлы в source
snapshots не являются новыми методами: это зафиксированные версии кода запусков.

`outputs/` в архиве **облегчен**: сохранены текстовые протоколы, метрики, конфиги,
сводки и код, но исключены бинарные выборки и состояния моделей. Наличие
`completed.json` означает исторически завершенный запуск, а не наличие всех
его артефактов в этом архиве. Старые абсолютные пути и PID в записях являются
историческими данными, а не текущими адресами или действующими процессами.

## 3. Что именно включено и исключено

Включены:

- текущий PDF и полный набор TeX/BibTeX/style/figure-файлов для его сборки;
- предыдущая редакция PDF и ее доступные TeX-исходники, отдельно от текущей;
- собственный код всех обнаруженных экспериментальных веток;
- необходимый авторский исходный код бейслайнов и его notices/license-файлы;
- тесты, конфиги, планы, обработчики метрик, графиков и диагностик;
- неизмененные зафиксированные исходники выполненных запусков;
- легкие JSON/CSV/Markdown-записи результатов и избранные готовые иллюстрации;
- версии зависимостей, сведения о происхождении кода и контрольные суммы.

Исключены:

- `.pt`, `.pth`, `.ckpt`, `.safetensors`, ONNX и иные файлы обученных весов;
- `.npy`, `.npz`, HDF5, меши/геометрические кэши и полные датасеты;
- `.conda`, `site-packages`, DLL/EXE, скачанные wheels и vendored Python-пакеты;
- `.git`, кэши, временные сборки, повторные распаковки и вложенные ZIP-архивы;
- крупные тренировочные трассы, если они превышают лимит текстовой выгрузки;
- выводы notebook-ячеек: код notebook сохранен, outputs и execution counters
  очищены только в архивной копии и отмечены в MANIFEST.

Точный состав определяется `MANIFEST.json`, а не этим сокращенным перечнем.
Канонический исследовательский Python-код и frozen snapshots не переписываются
ради упаковки. Для источников с измененными notebook-выводами записаны обе SHA256.

## 4. Какие результаты действительно получены

| Блок | Код | Статус в статье |
| --- | --- | --- |
| Девять задач Sphere/Torus, контроль инициализации и objectives | `paper/experiments/` | 72 завершенных обучающих запуска; основной источник `controlled_independent` |
| MGM, RW-Flow-C, KGDM, RMF-MT | `paper/benchmarks2026/` | План 108 запусков, завершены 18, прерваны 2, не начаты 88 |
| Volcano, современные методы | тот же блок | Все четыре метода, три training seed, 50 000 G/field updates |
| Earthquake, современные методы | тот же блок | Только KGDM и RMF-MT завершены по трем seed; MGM сохранен на 40k/20k, без тестовых результатов |
| SE(3), Mug/Table | `paper/benchmarks2026/grasp/` | Все 9 обучений по 10k updates и 24 оценки завершены |
| Bunny/Spot, k=10/50/100 | `external/riemannian-fm/rmgm_run_mesh_bunny_spot.py` | Исторические одно-seed пилоты; projection-surrogate, не точные spectral paths |
| Maze/boundaries, SPD, hyperbolic, high-D tori | исторические скрипты в `external/riemannian-fm/` | Пилоты/диагностики; не полный авторский бенчмарк |
| Полные авторские RFM-бюджеты | `rmgm_official_full_queue.py` | Код очереди есть, но завершение всех миллион-шаговых запусков не установлено |
| Реальные большие EEG SPD | соответствующие авторские конфиги | Не воспроизведены; маленькая SPD-синтетика их не заменяет |

На Volcano у MGM нет наименьшего наблюдаемого среднего MMD. В SE(3) MGM лучше
данной одношаговой адаптации RMF-MT по pose MMD, но хуже многошагового RFM.
Средняя доля коллизий полного Franka-меша у MGM составляет около 71%.
Это не доказательство физически успешного или готового к исполнению захвата.

Не переносите удачные цифры ранних toy-примеров из исторических README на
основные таблицы статьи. Протоколы, генераторы, kernels, бюджеты и выбор
чекпойнтов в этих блоках различаются.

## 5. Как устроен MGM

Генератор `G(z)` сразу возвращает точку допустимого многообразия. Critic нужен
только при обучении. На inference не интегрируется обученный flow и не вызывается
critic: используется один вызов генератора плюс геометрическая output-map.

Для пары generated/data endpoints строится геометрическая кривая `gamma(s)`.
Берутся `t ~ Uniform(0, 1/2)` и скрытый `B ~ Bernoulli(1/2)`, после чего
`s = t + B(1 - 2t)`. Critic видит `(t, gamma(s))`, но не B и не endpoints.
Target равен **настоящей производной кривой по s**, а не обязательно хорде
между endpoints и не производной сложной функции по t.

Critic минимизирует средний квадрат ошибки в римановой метрике:

```text
L_C = E[ ||f(t, gamma(s)) - d gamma(s)/ds||_g^2 ].
```

Генератор при замороженных параметрах critic минимизирует вариациональную форму:

```text
L_G = E[ 2 <f, d gamma(s)/ds>_g - ||f||_g^2 ].
```

Для generator step нужны производные через observation, velocity и зависящую
от точки метрику. Минимизировать по генератору просто critic regression нельзя:
отброшенный при обучении critic член зависит от generated endpoints.

В population-постановке reversible finite-energy paths дают определенную
симметричную discrepancy и bound на W1. Это **не** теорема о сходимости конечного
нейросетевого min-max обучения и не сертификат качества по малому train loss.

Геометрические варианты:

- sphere: ограничение нормы, great-circle path, касательный critic;
- torus: periodic angles и wrapped displacement; в новых версиях periodic features;
- noisy torus: кривая с `sin(pi*s)`-шумом и полной производной шума;
- SPD: Cholesky/output-constraint и выбранная матричная геометрия;
- hyperbolic: ball constraint и conformal metric, с численными оговорками;
- mesh: nearest-mesh projection либо sphere-map; исторический surrogate не
  объявляется точной intrinsic path-реализацией;
- SE(3): product SO(3) x R3, допустимые вращения и mesh-normalized translations.

## 6. Навигация по исходникам

| Путь | Что смотреть |
| --- | --- |
| `external/riemannian-fm/manifm/rmgm/generators.py` | исторические constrained MLP/GMM, sphere-map и surface-параметризации |
| `external/riemannian-fm/manifm/rmgm/geometry.py` | интерполяции, hidden flip, геометрические targets |
| `external/riemannian-fm/manifm/rmgm/critics.py`, `losses.py` | касательные critics и вариационные objectives |
| `paper/experiments/controlled_suite.py` | фиксированный малый GMM/MLP/RFM протокол |
| `paper/experiments/independent_suite.py` | immutable-source rerun и независимые RNG streams |
| `paper/experiments/correct_sphere_metrics.py` | анализ sphere NN/coverage в нормализованном float64 |
| `paper/experiments/mmd_control.py` | parameterization-matched direct-MMD objective control |
| `paper/experiments/stabilization*.py` | post-hoc validation-only optimization diagnostics |
| `paper/noisy_torus_study.py`, `verify_noisy_torus.py` | randomized torus path с настоящей velocity |
| `paper/benchmarks2026/common.py` | новые геометрии и общий unfitted point generator |
| `paper/benchmarks2026/mgm.py` | MGM, periodic critic, warmup, несколько C steps |
| `paper/benchmarks2026/rwflow.py` | chordal-cost RW-Flow-C и finite Sinkhorn solver |
| `paper/benchmarks2026/kgdm.py` | pinned spherical author field и torus adaptation |
| `paper/benchmarks2026/meanflow.py` | source-derived native two-time TMLP/JVP/PCGrad |
| `paper/benchmarks2026/worker.py` | обучение одного job, RNG, checkpoint selection, test scoring |
| `paper/benchmarks2026/run_jobs.py`, `gpu_guard.py` | bounded queue и проверки занятости GPU |
| `paper/benchmarks2026/select_configs.py`, `make_*plan.py` | validation-only выбор и формирование планов |
| `paper/benchmarks2026/plots.py`, `partial_report.py` | CPU-рисунки и честный неполный отчет |
| `paper/benchmarks2026/grasp/data.py`, `geometry.py` | ACRONYM, split/hash audits, SO(3) x R3 paths |
| `paper/benchmarks2026/grasp/models.py`, `source_mt.py` | conditional models и pinned RMF-MT loss algebra |
| `paper/benchmarks2026/grasp/evaluation.py` | pose MMD, static proxy и отдельные full-mesh collisions |
| `paper/benchmarks2026/grasp/pilot.py`, `cli.py` | последовательный пилот и явные команды |
| `paper/assemble_current.py` | таблицы и новые figure-assets из сохраненных результатов |

Полный автоматически сформированный список Python-исходников и CLI-аргументов
находится в [SOURCE_INDEX.md](docs/SOURCE_INDEX.md). Это статический индекс:
ради его построения скрипты не импортируются и не выполняются.

## 7. Окружение

Наблюдавшаяся среда: Windows, Python 3.10.20, PyTorch 2.5.1/CUDA 12.1,
одна RTX 4090. Наблюдавшиеся версии всех установленных пакетов сохранены в
`environment/observed_versions.json`; это справка, а не универсальный lockfile.

Пример подготовки отдельного окружения, выполняемый пользователем:

```powershell
conda create -n mgm-research python=3.10
conda activate mgm-research
$env:PYTHONUTF8 = '1'
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environment/requirements-core-reference.txt
```

Torch устанавливается отдельно для нужной платформы/CUDA. CPU-вариант подходит
для большинства unit tests, но не для исторических GPU-only trainers. Базовый
requirements описывает наблюдавшиеся прямые зависимости, а не проверенную
с нуля установку на любой ОС. Авторы исходного RFM использовали другую среду:
ее `environment.yml` сохранен как отдельная историческая спецификация.

Для старого RFM дополнительно нужны его optional dependencies, перечисленные
в `external/riemannian-fm/environment.yml` и `setup.py`. Не подменяйте молча
Python 3.10/PyTorch 2.5 современной версии пакета произвольным старым conda YAML.

Для SE(3) дополнительные compiled wheels устанавливаются отдельно:

```powershell
python -m pip install --target external/acronym-data/python-deps -r paper/benchmarks2026/grasp/requirements-isolated.txt
```

Сохраненный код автоматически добавляет этот каталог в sys.path. Сам каталог
с установленными пакетами в архив не включен. HDF5, RoMa, FCL и Rtree нужны для
полного grasp-пайплайна; collision backend зависит от платформы.

## 8. Проверки без обучения

Из корня распакованного проекта:

```powershell
python tools/verify_bundle.py
python tools/summarize_saved_metrics.py --check
$env:CUDA_VISIBLE_DEVICES = '-1'
python -B tools/run_cpu_tests.py --suite all
```

Первые две команды используют стандартную библиотеку Python и включенные
текстовые артефакты. Unit tests требуют установленных зависимостей. Архивный
runner оставляет исходные tests без изменений и явно пропускает одну проверку
Git checkout без `.git`, а также три data-dependent проверки при отсутствии
Fire CSV и Volcano NPZ. При восстановлении этих файлов проверки выполняются.
CUDA-only тесты graph solver без GPU должны пропускаться, а не выдавать фиктивный PASS.
Прямой `unittest discover` тоже возможен, но без Git metadata его
checkout-only test завершится ошибкой; это не ошибка loss/geometry. Полный
набор grasp data audits требует исключенных H5/мешей и поэтому не
объявляется доступным сразу после распаковки.

Не запускайте все `.py` подряд: часть исторических launcher-файлов при прямом
вызове начинает обучение или detached очередь. `--help` также не заменяет
статический просмотр неизвестного launcher. Проверка архива ничего такого
не делает.

## 9. Сборка статьи

Текущая TeX-рукопись находится в `paper/`, а не в каталоге исторической версии.
Все необходимые figure-PDF уже включены. GPU, datasets и checkpoints для
компиляции **не требуются**.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File paper/build.ps1
```

Скрипт использует MiKTeX по пути `C:\Program Files\MiKTeX\miktex\bin\x64`:
PDFLaTeX, BibTeX и повторные проходы. Если MiKTeX установлен иначе, измените
только `miktexBin` в своей распакованной копии. На другой TeX-системе можно
выполнить `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` из `paper/`.

Официальный ICML 2026 style сохранен без изменения. Формат не означает, что
работа была подана или принята на ICML. Авторы оставлены анонимными.

`paper/assemble_current.py`, `update_results.py` и генераторы рисунков не нужны
для обычной сборки PDF. Повторная генерация scientific figures требует
исключенных sample arrays, dataset caches и иногда checkpoint-файлов.

## 10. Данные и запуск современных сравнений

Прочитайте [DATA_AND_ASSETS.md](docs/DATA_AND_ASSETS.md). Нужны исходные данные,
исторические split indices и fixed target/reference caches. Одинаковые имена
датасетов без тех же splits не означают точного повторения чисел статьи.

`paper/benchmarks2026/data_metrics.py` использует подготовленные записи старого
контролируемого блока. Он не скачивает автоматически все девять датасетов и
не восстанавливает из одного JSON исключенные split arrays.

Готовая выбранная конфигурация находится в
`outputs/comparison2026/selected_config_v1.json`, исходный план в
`outputs/comparison2026/plans/final_v1.json`.

**Не запускайте этот исторический план как resume из архива.** В нем есть
облегченные `completed.json`, но нет соответствующих весов. Для нового запуска
создайте другую фазу; helper только записывает план, без GPU:

```powershell
python tools/new_comparison_plan.py --phase final_reproduction_v1
```

После подготовки и проверки данных, только при явном намерении обучать:

```powershell
# TRAINING: использует CUDA и может работать много часов.
python -u -m paper.benchmarks2026.run_jobs --plan outputs/comparison2026/plans/final_reproduction_v1.json
```

Это новый запуск с исходной инициализацией, а не продолжение прерванных MGM
checkpoint. В плане сохранены 50k generator/field updates и seeds 0/1/2.
MGM дополнительно делает warmup и 5/10 critic steps. Одинаковый cap на G steps
не равен одинаковым FLOPs, числу всех optimizer updates или размеру моделей.

Только после **полного** нового обучения идут isolated profiling, final audit,
report и plots. Для неполной фазы нельзя использовать полный final report.
Известные ограничения существующих вспомогательных скриптов перечислены в
[EXPERIMENTS.md](docs/EXPERIMENTS.md), включая Windows path keys в полном audit.

## 11. Ранние контролируемые и mesh-эксперименты

Пошаговые исходные протоколы сохранены в `paper/experiments/README.md`.
Важно не смешивать `controlled/`, неудачную промежуточную попытку и финальную
`controlled_independent/`: таблицы статьи используют последнюю.

Independent rerun требует сначала получить исходные 18 MLP objective-control
запусков и данные, затем 54 повторных GMM/RFM запуска и анализ sphere-v2.
Простое копирование `completed.json` этого не заменяет. Точные условия и
команды перечислены в [карте экспериментов](docs/EXPERIMENTS.md).

Исторические скрипты запускаются из `external/riemannian-fm/`. Например:

```powershell
Set-Location external/riemannian-fm
python rmgm_run_extra_geometries.py --section all --profile pilot --format markdown
```

Без `--run` этот helper печатает команды. У некоторых старых wrappers default
conda path привязан к прежней машине; передавайте `--conda`/`--env` или вызывайте
конкретный trainer через свое активное окружение. Для Bunny/Spot сохранены
`rmgm_prepare_mesh_data.py`, `rmgm_run_mesh_bunny_spot.py`,
`rmgm_mesh_compare.py` и `rmgm_mesh_sphere_map_notes.py`.

Параметр target k=10/50/100 обозначает выбранную eigenfunction target density,
а не автоматически число eigenfunctions для RFM path. Sphere-map не считается
биекцией лишь на основании топологии объекта. Maze с границами не подменяется
закрытой сфероподобной поверхностью.

## 12. SE(3)-пилот

Это отдельный **within-object held-out-pose** протокол на двух известных
публичных объектах Mug/Table, не полный ACRONYM и не unseen-object test.
Исторические positive FleX labels выбирают source poses, но не размечают
новые generated poses как физически успешные.

Подробности: `paper/benchmarks2026/grasp/README.md` и
`outputs/comparison2026/grasp/report.md`.

Чтобы подготовить данные заново, сначала отдельно получить разрешенные
публичные assets и зарегистрировать их hashes/license receipt. CLI содержит
`fetch-public`, `prepare`, `test`, `pilot-plan`, `pilot-check`, `train`,
`evaluate`, `visualize`, `pilot-run`. Загрузка данных не выполняется упаковщиком.

```powershell
python -m paper.benchmarks2026.grasp.cli fetch-public
python -m paper.benchmarks2026.grasp.cli prepare --grasps-root external/acronym-data/public/data/examples/grasps --mesh-root external/acronym-data/public/data/examples --out outputs/comparison2026/grasp/data/public_within_object --license-record external/acronym-data/public/download_manifest.json --smoke-only --protocol within_object_heldout_pose
```

Если архивная metadata уже занимает output-путь, используйте новый каталог и
создайте новый pilot plan. Исторический plan содержит source/data hash guards
и пути к отсутствующим в архиве checkpoint. Не обходите guards ради запуска.
`pilot-run --gpu-release` является обучающей командой и не выполняется
проверками архива.

## 13. Как читать метрики

- Sphere/torus: biased Gaussian embedded kernel MMD squared, diagonals included;
  train-only bandwidth. В таблицах обычно умножено на 1000.
- SE(3): unbiased fixed multiband product-kernel pose MMD, другой estimator;
  может быть отрицательным. Его числа не сопоставляются напрямую с первым.
- Minimum matching distance также иногда сокращают MMD, но это NN-метрика,
  а не kernel MMD; в коде имена разнесены.
- Mean/SD: сначала усреднение evaluation draws внутри training seed, затем
  sample SD между тремя independently initialized training seeds.
- SE(3): сначала equal-object mean внутри seed, затем SD между seeds.
  Шесть object/seed combinations не являются шестью training replicates.
- 1-NNA: ориентир около 0.5, не правило «чем меньше, тем лучше».
- Static proxy, full-Franka collision и физический успех различны. Ни один
  из первых двух не заменяет динамический grasp success.
- Historical row splits не проверяют unseen proteins, молекулярные группы
  или временное forecasting. Ранее просмотренный test не становится новым
  слепым benchmark после фиксации плана.

## 14. Источники, лицензии и границы использования

[THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) перечисляет локальные
репозитории, revisions и найденные license/notices. RFM имеет CC BY-NC 4.0,
KGDM содержит MIT license. В сохраненном RMF-репозитории отдельный LICENSE не
обнаружен; присутствуют в том числе Meta/all-rights-reserved notices. Упаковка
не добавляет разрешения на его публичное распространение или коммерческое
использование. Это рабочий архив имеющегося исследовательского материала,
а не заявление о единой open-source лицензии всего пакета.

Файлы авторских repos, copied adapters и собственные реализации должны быть
атрибутированы отдельно. Пакет не заявляется анонимизированным supplementary
submission: исторические README/JSON сохраняют сведения о рабочей машине.
Перед внешней публикацией отдельно проверьте права и анонимизацию.

## 15. Что проверено при упаковке

В распакованной копии выполнены 144 CPU unit tests: 138 прошли, 6 явно
пропущены (2 CUDA, 1 Git metadata, 3 исключенных data fixtures). Ошибок нет,
CUDA context не инициализировался. Полный отчет включен в
`docs/SMOKE_TESTS.json`. Новое окружение с нуля при этом не устанавливалось.

Фактический отчет находится в `docs/PACKAGE_CHECKS.json` и соседнем с ZIP
внешнем отчете проверки. Проверяются CRC, SHA256 каждого включенного файла,
полнота заявленного набора исходников, отсутствие запрещенных бинарных
артефактов и совпадение PDF с исходной проверенной редакцией.

Выполненные unit tests, сравнение сохраненных агрегатов и ограничения проверки
перечисляются по факту; нельзя считать сборку архива новым воспроизведением
обучения, collision simulation или convergence audit. Исключенные веса и данные
на исходной машине не удаляются и не изменяются. Очередь остается остановленной.
