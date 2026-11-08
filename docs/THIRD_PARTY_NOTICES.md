# Сторонний код и ограничения

Это описание найденных локальных license/notices, а не новая единая лицензия
для всего архива. Собственные изменения, скопированные определения и авторские
репозитории не объявляются автоматически MIT или public domain.

| Каталог | Происхождение | Найденные условия |
| --- | --- | --- |
| `external/riemannian-fm/` | facebookresearch/riemannian-fm и локальные MGM additions | `LICENSE`: CC BY-NC 4.0; copyright/notices сохранены |
| `external/riemannian-fm-official-clean/` | отдельная авторская копия RFM | `LICENSE`: CC BY-NC 4.0; не путать с локально дополненной версией |
| `external/kgdm-author/` | jcarrascopollo/kgd_thesis | `LICENSE`: MIT; pinned field не изменен |
| `external/riemannian-meanflow-author/` | haolsun/Riemannian_MeanFlow | отдельный LICENSE не найден; присутствуют в том числе Meta/all-rights-reserved notices |
| `external/acronym-data/public/` | NVlabs/acronym | repository code MIT, dataset assets CC BY-NC 4.0; raw assets в архив не включены |
| `paper/icml2026.*`, `paper/vendor/icml2026/` | официальный ICML 2026 template | сохранены исходные notices; style не изменялся |
| `paper/vendor/ne_110m_land.geojson` | Natural Earth | public-domain geographic context, не dataset target |

Точные Git revisions и наличие локальных изменений регистрируются в
`docs/UPSTREAM_PROVENANCE.json`. `.git` не включается. Сам факт включения
доступной рабочей копии в личный исследовательский архив не добавляет прав
на ее публичное распространение или коммерческое использование.

## Зафиксированные методы

- KGDM author revision: `5860d212a1e53b53a499e251f2d087fcf0a5fc00`.
  `earth/drift_core.py` проверяется по LF-normalized SHA256
  `39008dd140a304851dc200216d035c714fe3264c9319ee28bb534caefbf0a5d0`.
- RMF author revision: `45d96bcd09a271ea3a770b2014865dabbac9eeeb`.
  Source-derived TMLP/loss сохраняет notices. SE(3) adapter отдельно проверяет
  hashes `grasp/src/models/lightning.py` и `pcgrad.py`.
- ACRONYM public revision: `d02f58fb3a556c3b719976e206cce0dce0103199`.
- RW-Flow-C реализован по уравнениям статьи; он не называется проверенным
  авторским репозиторием RW-Flow.

## Основные первоисточники

- RFM: https://arxiv.org/abs/2302.03660
- MGM: https://arxiv.org/abs/2605.29920v2
- RW-Flow: https://arxiv.org/abs/2609.39271v2
- KGDM: https://arxiv.org/abs/2605.10727
- RMF Zhong et al.: https://arxiv.org/abs/2603.10718v3
- ACRONYM: https://arxiv.org/abs/2011.09584
- ICML template/rules: https://icml.cc/Conferences/2026/AuthorInstructions

Статья содержит полную библиографию. Не удаляйте notices из авторских файлов.
Рабочий пакет не является прошедшим лицензионную проверку публичным release
или готовым анонимизированным supplementary submission.
