# Журнал лабораторной работы 2 (дата, версии, команды, решения). Вариант 15

Дата: 2026-10-07 16:30:15 UTC

## Версии

python 3.11.5
pip 23.2.1

## Команды

python src\lab02_make_variant.py --variant 15 --out artifacts\v15
python src\lab02_audit.py --manifest artifacts\v15\manifest.csv --policy artifacts\v15\policy.json --out reports\audit_result
python src\lab02_model_card.py --passport ..\\..\\лаб1.1\\lab01\\artifacts\\v15\\passport.json --out artifacts\\model_card.md
python src\lab02_risks.py --template artifacts\risk_register.csv
python src\lab02_risks.py --csv artifacts\risk_register.csv --out reports\risk_ranking.md

## Решения

| Объект | Находки | Решение | Обоснование |
|--------|---------|---------|-------------|
| item01 | R4 | Исключить или заменить на CC0/CC-BY/CC-BY-SA; при сохранении — использовать без изменений | policy.json содержит "derivatives": true, значит сценарий требует возможности создавать производные, но лицензия CC-BY-ND-4.0 прямо запрещает создание производных |
| item04 | R4 | Исключить или заменить на CC0/CC-BY/CC-BY-SA; при сохранении — использовать без изменений | policy.json содержит "derivatives": true, значит сценарий требует возможности создавать производные, но лицензия CC-BY-ND-4.0 прямо запрещает создание производных |
| item08 | R10 | Найти правообладателя и заполнить поле; при неудаче — исключить объект | Поле rights_holder пустое - это нарушение условия лицензии |
| item12 | R4 | Исключить или заменить на CC0/CC-BY/CC-BY-SA; при сохранении — использовать без изменений | policy.json содержит "derivatives": true, значит сценарий требует возможности создавать производные, но лицензия CC-BY-ND-4.0 прямо запрещает создание производных |