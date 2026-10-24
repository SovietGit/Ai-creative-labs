# Протокол проверки воспроизводимости ЛР №1

## 1. Предмет проверки

- Дисциплина: «Искусственный интеллект в креативных технологиях».
- ЛР №1: «Запуск открытой модели text-to-image и фиксация воспроизводимого результата».
- Дата фактической проверки: 23.09.2026.
- Модель: `stabilityai/sd-turbo`.
- Закреплённая ревизия: `b261bac6fd2cf515557d5d0707481eafa0485ec2`.
- Режим: локальный инференс на CPU, без внешнего генеративного сервиса.

## 2. Информация о варианте

Вариант 15: иллюстрация к статье о цифровых архивах.

### Контекст варианта

- **Тема:** иллюстрация для материала о цифровых архивах.
- **Изменяемый фактор:** метафора поиска и сохранения данных.
- **Обязательная проверка:** валидация JSON-манифеста.
- **Артефакты:** иллюстрация и README.
- **Критерий результата:** метафора связана с темой цифрового архива.
- **Ограничение:** не показывать персональные документы, реальных лиц и логотипы.

## 3. Цель и постановка задачи

**Цель**: Получить один воспроизводимый результат text-to-image и доказать
происхождение каждого артефакта.

**Задачи**:
1. Создать структуру проекта и зафиксировать environment.txt. Контрольная точка: версии
читаются, секреты и кэш весов отсутствуют.
2. Выбрать вариант и сформировать безопасный prompt с предметом, композицией,
палитрой и ограничениями. Контрольная точка: нет реальных лиц, логотипов и закрытых
данных.
3. Запустить модель с закреплённой ревизией. Контрольная точка: PNG и JSON созданы одним
запуском.
4. Повторить запуск без изменения условий. Контрольная точка: сравните SHA-256; при
расхождении зафиксируйте диагностику.
5. Воспроизвести типовую ошибку на безопасной копии кода, описать симптом и исправление.
6. Собрать отчёт и архив без виртуального окружения и кэша модели.

## 4. Параметры

- Prompt: `abstract illustration for a digital archive concept, a glowing network of interconnected data nodes and light streams, deep blue and soft cyan palette, clean minimal composition, no text, no logos, no people`
- Seed: `20260917`
- Число шагов: `1`
- Guidance scale: `0.0`
- Размер: `512 × 512`
- Тип данных: `torch.float32`
- Детерминированные алгоритмы PyTorch: `True`
- Число потоков PyTorch: `1`

## 5. Фактическая среда

- Python: `3.11.5`
- PyTorch: `2.14.0+cu126`
- Torchvision: `0.29.0+cu126`
- Diffusers: `0.31.0`
- Transformers: `4.57.6`
- Accelerate: `1.15.0`
- Safetensors: `0.8.0`
- Hugging Face Hub: `0.36.2`

Полный список пакетов сохранён в `reports/environment.txt`.

## 6. Команда запуска и исходный код

```bash
export HF_HOME="$PWD/cache/huggingface"
.venv/bin/python src/generate_once.py 2>&1 | tee reports/run_001.log
```

### Исходный код

Исходный код находится в src/generat_once.py

```python
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.metadata as md, json, platform, time
import torch
from diffusers import AutoPipelineForText2Image

MODEL_ID = "stabilityai/sd-turbo"
REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
PROMPT = (
    "abstract illustration for a digital archive concept, "
    "a glowing network of interconnected data nodes and light streams, "
    "deep blue and soft cyan palette, "
    "clean minimal composition, no text, no logos, no people"
)
SEED, STEPS, GUIDANCE, SIZE = 20260917, 1, 0.0, 512

out = Path("artifacts/run_001")
out.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
load_args = dict(revision=REVISION, use_safetensors=True, torch_dtype=dtype)
if device == "cuda":
    load_args["variant"] = "fp16"

started = time.perf_counter()
pipe = AutoPipelineForText2Image.from_pretrained(MODEL_ID, **load_args)
pipe = pipe.to(device)
print("Device:", device)
print("CUDA available:", torch.cuda.is_available())
generator = torch.Generator(device="cpu").manual_seed(SEED)
image = pipe(
    prompt=PROMPT,
    num_inference_steps=STEPS,
    guidance_scale=GUIDANCE,
    height=SIZE,
    width=SIZE,
    generator=generator,
).images[0]
elapsed = time.perf_counter() - started

image_path = out / "result.png"
image.save(image_path)
sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

packages = {name: md.version(name) for name in ["torch", "diffusers", "transformers", "accelerate", "safetensors"]}
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "model_id": MODEL_ID,
    "revision": REVISION,
    "prompt": PROMPT,
    "seed": SEED,
    "steps": STEPS,
    "guidance_scale": GUIDANCE,
    "height": SIZE,
    "width": SIZE,
    "device": device,
    "dtype": str(dtype),
    "platform": platform.platform(),
    "packages": packages,
    "elapsed_seconds_measured": elapsed,
    "artifact": str(image_path),
    "sha256": sha256,
}
(out / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
```

## 7. Результаты двух запусков

После двух запусков были созданы папки run_001/ и run_002/ и помещены в artifacts/ с содержимым:
- manifest.json (результат запуска);
- result.png (итоговое изображение).

### Содержимое manifest.json в папке run_001/:

```text
{
  "created_utc": "2026-09-23T12:04:41.620107+00:00",
  "model_id": "stabilityai/sd-turbo",
  "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
  "prompt": "abstract illustration for a digital archive concept, a glowing network of interconnected data nodes and light streams, deep blue and soft cyan palette, clean minimal composition, no text, no logos, no people",
  "seed": 20260917,
  "steps": 1,
  "guidance_scale": 0.0,
  "height": 512,
  "width": 512,
  "device": "cpu",
  "dtype": "torch.float32",
  "platform": "Windows-10-10.0.26200-SP0",
  "packages": {
    "torch": "2.14.0+cu126",
    "diffusers": "0.31.0",
    "transformers": "4.57.6",
    "accelerate": "1.15.0",
    "safetensors": "0.8.0"
  },
  "elapsed_seconds_measured": 18.31029999999737,
  "artifact": "artifacts\\run_001\\result.png",
  "sha256": "19c82891b6dede5a5e62eb8ef72892bd682dc6a6eee5294820a15cc5d318811f"
}
```

### Итоговое изображение после первого запуска
![Результат запуска 1](artifacts/run_001/result.png)

### Содержимое manifest.json в папке run_002/:

```text
{
  "created_utc": "2026-09-23T13:06:49.606057+00:00",
  "model_id": "stabilityai/sd-turbo",
  "revision": "b261bac6fd2cf515557d5d0707481eafa0485ec2",
  "prompt": "abstract illustration for a digital archive concept, a glowing network of interconnected data nodes and light streams, deep blue and soft cyan palette, clean minimal composition, no text, no logos, no people",
  "seed": 20260917,
  "steps": 1,
  "guidance_scale": 0.0,
  "height": 512,
  "width": 512,
  "device": "cpu",
  "dtype": "torch.float32",
  "platform": "Windows-10-10.0.26200-SP0",
  "packages": {
    "torch": "2.14.0+cu126",
    "diffusers": "0.31.0",
    "transformers": "4.57.6",
    "accelerate": "1.15.0",
    "safetensors": "0.8.0"
  },
  "elapsed_seconds_measured": 12.028877999997349,
  "artifact": "artifacts\\run_002\\result.png",
  "sha256": "19c82891b6dede5a5e62eb8ef72892bd682dc6a6eee5294820a15cc5d318811f"
}
```

### Итоговое изобаржение после второго запуска
![Результат запуска 2](artifacts/run_002/result.png)

### Содержимое reports/sha256_comparison.json

```text
{
  "checked_utc": "2026-09-23T15:44:21.451158+00:00",
  "run_001_sha256": "19c82891b6dede5a5e62eb8ef72892bd682dc6a6eee5294820a15cc5d318811f",
  "run_002_sha256": "19c82891b6dede5a5e62eb8ef72892bd682dc6a6eee5294820a15cc5d318811f",
  "exact_sha256_match": true,
  "same_image_size": true,
  "same_image_mode": true,
  "interpretation": "PASS: два запуска в одной зафиксированной CPU-среде побитово совпали."
}
```

### Итоговая таблица сравнения итогов двух запусков

| Проверка | Запуск 1 | Запуск 2 |
| --- | ---: | ---: |
| Время генерации, с | 1027.75 | 12.02 |
| Размер файла, байт | 542128 | 542128 |
| Размер изображения | 512 × 512 | 512 × 512 |
| Цветовой режим | RGB | RGB |
| SHA-256 | `19c82891b6dede5a5e62eb8ef72892bd682dc6a6eee5294820a15cc5d318811f` | `19c82891b6dede5a5e62eb8ef72892bd682dc6a6eee5294820a15cc5d318811f` |

Побитовое сравнение: **PASS**.

Два PNG полностью совпадают; одинаковый SHA-256 подтверждает повторяемость результата
в пределах данной зафиксированной CPU-среды.

### Аномально долгое время генерации при первом запуске

Первый запуск занял 1027.75 секунд, второй - 12.02 секунд. Разница обусловлена загрузкой весов модели с Hugging Face при первом запуске. При повторном запуске веса берутся из локального кэша, поэтому время сокращается до времени генерации. 

## 8. Выявленное расхождение методических указаний

Первоначальная зависимость `diffusers==0.40.0` вместе с `transformers>=4.51,<5` не устанавливается: Diffusers 0.40.0 требует `huggingface-hub>=1.23`. 

Для фактического запуска было принято решение о понижении версии `diffusers` до `0.31.0`. Методические указания следует актуализировать.

## 9. Безопасность и лицензирование

Использован нейтральный prompt без персональных данных, реальных лиц, товарных
знаков и закрытых материалов.

Загруженная конфигурация pipeline не содержит safety checker. Это допустимо только
для контролируемого учебного эксперимента с безопасным prompt; для открытого сервиса
нужен отдельный фильтр и политика модерации.

Данный проект использует модель `stabilityai/sd-turbo`.

### Лицензия

Модель распространяется под лицензией **Stability AI Community License**.
Использование бесплатно для организаций и лиц с годовым доходом менее $1,000,000 (USD).
При превышении этого порога требуется Enterprise License.

Полный текст лицензии: https://stability.ai/license

### Acceptable Use Policy (AUP)

Использование модели должно соответствовать **Acceptable Use Policy** Stability AI.
AUP запрещает, в частности, создание контента, связанного с насилием, эксплуатацией несовершеннолетних, сексуально откровенного контента, дезинформации и нарушений прав третьих лиц.

Полный текст AUP: https://stability.ai/use-policy

## 10. Границы вывода

Результат подтверждает точное совпадение двух запусков в одной среде. Он не доказывает
побитовую воспроизводимость между CPU и GPU, различными ОС, версиями библиотек или
архитектурами ускорителей.

## 11. Итог

Статус проверки: **PASS**.

Файл `reports/sha256_comparison.json` содержит
`"exact_sha256_match": true`, что подтверждает побитовое совпадение двух запусков
внутри одной зафиксированной среды, но не обещает совпадение между разными
устройствами, версиями PyTorch или платформами.

Артефакты позволяют проверить параметры, версии, два результата, журналы запусков и
контрольные суммы. Кэш весов и виртуальное окружение намеренно не включены в архив.

Сформированы:
- reports/sha256_comparison.json (сравнение двух запусков);
- artifacts/run_001/manifest.json (результат запуска 1);
- artifacts/run_001/result.png (итоговое изображение запуска 1);
- artifacts/run_002/manifest.json (результат запуска 2);
- artifacts/run_002/result.png (итоговое изображение запуска 2);

## 12. Намеренно введенная типовая ошибка

Ошибка: заменить generator=torch.Generator(device="cpu").manual_seed(SEED) на
generator=torch.Generator(device="cuda").manual_seed(SEED) и затем повторить запуск на CPUмашине.

Симптом: исключение о недоступном CUDA или несовместимом устройстве генератора.

Диагностика: вывести device и torch.cuda.is_available(), проверить строку создания Generator.

Исправление: создать CPU Generator, как в эталонном коде, либо использовать устройство только
после проверки доступности. После исправления удалить неполный каталог запуска и выполнить
команду заново

Результат воспроизведения типовой ошибки находится в `reports/run_error.log`.
