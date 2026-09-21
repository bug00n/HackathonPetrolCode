# Разработка и воспроизведение

Для показа без настройки скачайте [готовую Windows-сборку](https://github.com/bug00n/HackathonPetrolCode/releases/latest). Этот документ нужен только при работе с исходниками. Команды PowerShell выполняются из корня репозитория.

## 1. Исходники и Python

Нужны Git и Python 3.12 с Tkinter. Для исходных больших данных отдельно нужен Git LFS; модельные сценарии не требуют распаковки производственной телеметрии.

```powershell
git clone https://github.com/bug00n/HackathonPetrolCode.git
cd HackathonPetrolCode
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m source.main validate-stage0
.\.venv\Scripts\python.exe -m source.ui --page overview
```

Репозиторий закрытый: для clone и Releases требуется разрешённый доступ. Зависимости устанавливаются через сеть либо из заранее подготовленного wheelhouse. Далее вычисления локальные.

`validate-stage0` ожидает 8 конфигураций сценариев, 6 модельных fixtures и 170 известных тегов. В UI доступны шесть модельных сценариев. Отсутствие исторических ресурсов не должно подменяться случайной другой моделью.

## 2. История без переобучения

Скачайте и распакуйте **тот же релиз**, с тегом которого работаете. Из его корня перенесите в checkout с сохранением вложенности:

| Каталог в поставке | Назначение |
| --- | --- |
| `data/processed/66bdfcbb23b4` | Подготовленные данные организаторов |
| `artifacts/models/sulfur-upper-33efba3c3141` | Основной исторический прогноз |
| `artifacts/models/sulfur-v2-shadow-61f972d07181` | Исследовательский v2-прогноз |

Не заменяйте существующие экспериментальные каталоги: копируйте только отсутствующие закреплённые каталоги. Точные пути задаёт `config/release_manifest.json`. Модельные файлы доверенные, из поставки проекта; не загружайте произвольные pickle/joblib из неизвестных источников.

```powershell
.\.venv\Scripts\python.exe -m source.main history-interval `
  --dataset data/processed/66bdfcbb23b4 `
  --model artifacts/models/sulfur-upper-33efba3c3141 `
  --from 2025-06-01T12:00:00+03:00 `
  --to 2025-06-01T13:00:00+03:00
```

Результат содержит две точки, point/upper и причины ограничений; управляющие действия не разрешаются. Исторические этапы могут содержать старые ID — не используйте их вместо manifest.

## 3. Проверки

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m mypy source
.\.venv\Scripts\python.exe -m source.main acceptance --output reports/acceptance-local
```

Для `acceptance` выбирайте новый выходной каталог. Проверяются пять основных эпизодов; шестой `blend_tradeoff` проверяется regression-тестом. GitHub Actions запускает проверки на Windows/Python 3.12 и Linux/Python 3.11. На Linux для настоящего GUI-теста нужен дисплей, например `xvfb-run -a python -m pytest`.

## 4. Собрать переносимую Windows-версию

Сначала подготовьте окружение из раздела 1 и ресурсы из раздела 2. Выходные каталоги должны быть новыми. PowerShell-скрипты запускайте согласно политике вашей машины, не отключая защиту.

```powershell
.\scripts\build_windows.ps1 -BuildOutput D:\Neftekod-build
.\scripts\build_release.ps1 `
  -BinaryDirectory D:\Neftekod-build\dist\Neftekod `
  -Output D:\Neftekod-delivery -Archive
.\.venv\Scripts\python.exe scripts/verify_release.py D:\Neftekod-delivery.zip
```

Первый скрипт устанавливает изолированные зависимости сборки и создаёт EXE с runtime. Второй копирует исходники, документацию и закреплённые ресурсы, прогоняет acceptance и сам EXE из новой папки с PATH без Python. Затем формируются `RELEASE_MANIFEST.json`, `SHA256SUMS.txt` и ZIP. Сборка не должна публиковаться при ошибке проверки.

Для выпуска используйте чистый commit. EXE и большие подготовленные ресурсы размещайте в GitHub Release, а не в обычных Git-коммитах. Не меняйте бинарный файл существующего релиза без новой версии и новых сумм.

## 5. Подготовка данных и ML-исследования

Это отдельный разработческий маршрут, не шаг первого запуска:

```powershell
git lfs install
git lfs pull
.\.venv\Scripts\python.exe -m source.main prepare --materials materials --config config/runtime.toml
.\.venv\Scripts\python.exe -m source.main --help
```

Исходный `materials/data.rar` хранится в Git LFS; для чтения RAR нужен совместимый `tar`. Подготовка выводит фактический dataset ID. Новое обучение не обязано создать ID или байты закреплённого артефакта. Команды `train`, `evaluate`, `train-v2-shadow` и исследовательские команды описаны в CLI и документах этапов 6–10. Validation после финального refit — диагностика; для независимой оценки используйте test. Никакая прогнозная метрика не включает `supports_actions`.

Временные файлы, `.venv`, журналы, подготовленные данные и модели игнорируются Git. Архивные доказательства конкретной поставки находятся в `release/evidence/`; они не заменяют новый прогон после изменения вычислительного кода.
