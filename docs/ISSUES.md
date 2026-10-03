# Proctoring System — Issues & Fix Plan

## Проблемы по приоритету

### 🔴 Высокий приоритет (прямо влияют на FP rate)

- [x] **#1 — `conf=0.05` слишком низкий** → исправлено: `0.30`
  - Файл: `project_files/newArch.py`
  - Шумные keypoints с низкой уверенностью создают случайные смещения носа → FP

- [x] **#2 — Штраф по Y-оси (вперёд/назад) = FP от письма** → исправлено
  - Файл: `project_files/newArch.py`
  - Теперь штрафуется только боковое (X) смещение носа; наклон вперёд для письма (Y) подавляется
  - Новый параметр: `POSE_DEPTH_SUPPRESS_RATIO` (default 1.5)

- [x] **#3 — Угловая метрика (`angle_diff`) неверна для потолочной камеры** → отключена
  - Файл: `project_files/newArch.py`
  - `POSE_ANGLE_WEIGHT` изменён с `1.0` на `0.0` по умолчанию
  - Включить обратно (env var) только при фронтальной камере

- [x] **#4 — Foreshortening: плечи сужаются под углом камеры** → исправлено
  - Файл: `project_files/newArch.py`
  - Добавлен `effective_base_radius` с поправкой по Y-позиции bbox в кадре
  - Новые параметры: `POSE_FORESHORTENING_STRENGTH` (default 0.4), `POSE_FRAME_HEIGHT` (default 720)

---

### 🟡 Средний приоритет (производительность и корректность)

- [x] **#7 — Калибровка 10с фиксировала позу "голова вниз"** → исправлено: `30с`
  - Файл: `project_files/newArch.py`
  - `POSE_CALIB_S` default: `10.0` → `30.0`
  - Хардкод `10.0` при возврате из "away" тоже заменён на `pose_calib_s`

- [x] **#8 — `auto_tune_from_debug_csv` не использовался** → задокументировано ниже

---

---

## Как использовать auto-tune (снижение FP без ручного перебора)

### Шаг 1 — Собрать данные нормального поведения

Запустить систему на записи экзамена, где нет списывания (или первые 30 минут нормального экзамена).

```bash
set POSE_DEBUG=1
python worker.py
```

Это создаёт файл `evidence_folder/pose_debug.csv` с колонками:
```
ts, cid, person_key, dist, lateral_dev, depth_dev, dist_excess, abs_angle, angle_excess_norm, combined_excess, score_s
```

### Шаг 2 — Запустить авто-тюнинг

```bash
set POSE_AUTO_TUNE=1
python worker.py
```

Система прочтёт CSV и подберёт `base_radius`, `angle_base`, `pose_score_k` так, чтобы
ложных тревог было ≤ 0.5 в час (FAH ≤ 0.5). Результат сохраняется в
`evidence_folder/auto_tune_result.json`.

### Шаг 3 — Применить параметры

```bash
set POSE_BASE_RADIUS=<base_radius_sugg из json>
set POSE_SCORE_K=<chosen_k из json>
python worker.py
```

### Параметры для тонкой настройки (env vars)

| Переменная | Default | Описание |
|---|---|---|
| `POSE_BASE_RADIUS` | `0.38` | Радиус "нормальной зоны" (больше = менее чувствительно) |
| `POSE_WARN_THRESHOLD_S` | `10.0` | Секунд подозрительного поведения до тревоги |
| `POSE_CALIB_S` | `30.0` | Секунд калибровки в начале |
| `POSE_DEPTH_SUPPRESS_RATIO` | `1.5` | При depth > lateral × ratio — подавить (поза письма) |
| `POSE_FORESHORTENING_STRENGTH` | `0.4` | Сила поправки на перспективу (0 = выкл) |
| `POSE_FRAME_HEIGHT` | `720.0` | Высота кадра для расчёта поправки |
| `POSE_ANGLE_WEIGHT` | `0.0` | Вес угловой метрики (0 = выкл, только для фронт. камер) |
| `POSE_SCORE_K` | `1.0` | Скорость накопления подозрительности |
| `POSE_DEBUG` | `0` | Писать CSV для анализа (1 = вкл) |
| `POSE_AUTO_TUNE` | `0` | Авто-тюнинг при старте (1 = вкл, нужен CSV) |
| `POSE_SAVE_CLIPS` | `0` | Сохранять кандидат-кадры для разметки |

---

## Порядок выполненных работ

1. ✅ Создан план (ISSUES.md)
2. ✅ #1 — Поднят conf: 0.05 → 0.30
3. ✅ #2 — Разделён lateral/depth штраф (только X ось)
4. ✅ #3 — Отключена угловая метрика для потолочных камер
5. ✅ #4 — Добавлена поправка на foreshortening по Y-позиции bbox
6. ✅ #5 — MAX_BATCH: 1 → 4
7. ✅ #6 — FPS: sleep(0.2) → sleep(0.04) (5→25 FPS)
8. ✅ #7 — Калибровка: 10с → 30с
9. ✅ #8 — Задокументирован процесс auto_tune
10. ✅ #9/#10 — Хардкод путей и shared memory
