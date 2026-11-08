# Данные, кэши и внешние assets

Архив содержит код подготовки, идентификаторы, конфиги и lightweight manifests,
но не полные исходные данные, обученные модели или бинарные generated samples.
Это намеренное ограничение объема. На исходной машине ничего не удаляется.

## 1. Где код ожидает файлы

| Данные | Ожидаемый путь | Что требуется |
| --- | --- | --- |
| Earth events | `external/riemannian-fm/data/earth_data/` | author-format files Volcano/Earthquake/Flood/Fire |
| Protein torsions | `external/riemannian-fm/data/top500/` | исходные molecular files и соответствующий preprocessing |
| RNA | `external/riemannian-fm/data/rna/` | исходный локальный preprocessing, 9473 rows в статье |
| Bunny/Spot/maze | `external/riemannian-fm/data/mesh/` | object geometry и target sample arrays |
| Locked controls | `outputs/paper_icml2026/controlled_independent/datasets/<name>/` | split indices, fixed targets, train references; arrays исключены |
| Modern comparison | `outputs/comparison2026/data/<name>/data.npz` | train/val/test и fixed target/reference arrays |
| ACRONYM public examples | `external/acronym-data/public/data/examples/` | matched Mug/Table meshes и grasp H5 |
| Franka collision mesh | `external/acronym-data/public/data/franka_gripper_collision_mesh.stl` | geometry для отдельной collision-метрики |
| Grasp prepared cache | `outputs/comparison2026/grasp/data/public_within_object/` | source indices, object clouds/poses, hash-consistent cache |

JSON manifests в этих каталогах могут присутствовать без бинарных файлов.
Такой каталог нельзя считать подготовленным только по наличию manifest.

## 2. Источники и подготовка

Исходный RFM: https://github.com/facebookresearch/riemannian-fm

Датасетные загрузчики находятся в `manifm/datasets.py`, исходный mesh synthesis
в `data/synthesize_mesh_data.py`, локальные wrappers в
`rmgm_prepare_torus_data.py` и `rmgm_prepare_mesh_data.py`.

Мolecular download/preprocessing использует RCSB. Повторная загрузка сегодня
может не дать byte-identical файлы историческому кэшу. Hash guards служат
для обнаружения этого отличия; не отключайте их, чтобы объявить данные теми же.

Для sphere/torus tests совпадение названия задачи не обеспечивает совпадения
row split, evaluation target, reference bandwidth или coverage radius. В статье
используется исторический data seed0 и сохраненные exact indices. Если эти
arrays недоступны, новый split является новым экспериментом, а не точным
воспроизведением старых чисел.

`paper/benchmarks2026/data_metrics.py` копирует и проверяет существующие locked
records предыдущего контролируемого блока. Это не универсальный downloader.

## 3. Поверхности

Для eigenfunction targets используются исходные `bunny_simp.obj`, `spot_simp.obj`
и создаваемые `*_eigfn*.npy`. Геометрия и spectral caches не являются весами
нейросети, но они тоже исключены как бинарные experimental assets.

Создание target arrays может быть тяжелым CPU/GPU preprocessing. Например,
`rmgm_prepare_mesh_data.py` по умолчанию использует 500000 eigenfunction samples
и upsampling. Это не команда мгновенной проверки.

Maze/boundary configurations требуют своих данных и boundary-aware assumptions.
Projection valid output не доказывает выполнение всех условий exact-path theorem.

## 4. ACRONYM

Источник: https://github.com/NVlabs/acronym

Сохраненный код имеет команду `fetch-public`, получающую только публичные
Mug/Table examples и фиксирующую raw asset/source hashes. Полный ShapeNetSem
доступ и права на его assets не считаются предоставленными этим архивом.
Большой grasp archive без соответствующих мешей не заменяет полноценный dataset.

Код `grasp/data.py` проверяет source H5 rows, positive-only source split,
масштаб mesh, cached transforms и source permutation. H5 translations уже в
метрах, поэтому scale применяется к mesh, а не второй раз к pose translation.

Metadata для two-object пилота: `outputs/comparison2026/grasp/data/` и
`outputs/comparison2026/grasp/pilot_plan_gpu_v2.json`. Первоначальный
`public_smoke` object-transfer split не является основным held-out-pose pilot.

## 5. Какие результаты читаются без этих файлов

Без весов и datasets доступны PDF/TeX, таблицы, отчеты и числовая reaggregation
в `tools/summarize_saved_metrics.py --check`. Возможны algebraic/unit tests
с synthetic fixtures при установленных зависимостях.

Недоступны без восстановления исключенных artifacts:

- повторный inference и sampling с сохраненной обученной моделью;
- продолжение прерванного optimizer/RNG state;
- independent kernel recomputation из raw samples;
- новый collision audit из pose arrays и meshes;
- повторная генерация всех иллюстраций из NPZ;
- полная `verify_independent.py`/full-phase audit.

Сохраненный PASS старого audit означает выполненную ранее проверку, а не
новое ее выполнение в урезанном архиве.

## 6. Переносимость путей и source guards

В рабочем коде преобладают project-relative paths. Исторические JSON и часть
старых wrappers содержат абсолютные Windows paths. Они сохранены ради
происхождения записей и не являются обещанием переносимости.

Для нового запуска создавайте новые manifests/plans через CLI и задавайте
свой output-root. Не редактируйте старый source snapshot, не меняйте hashes
задним числом и не копируйте веса от другой конфигурации/seed в якобы тот же job.
В архиве отсутствуют `.git` и checkpoints; это не мешает чтению кода, но исключает
автоматическое продолжение всех исторических запусков.
