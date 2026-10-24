# Паспорт по части Б ЛР 1.1 - Вариант 15

Дата: 2026-09-30 16:58:30 UTC

Версия: Engee 26.9.2-H1 v. 26.9.1.1

## Созданные папки и файлы

Папки: 
- "ИИ в КТ" (корневая папка), 
- "LR01" ("LR01" -> "ИИ в КТ").

Файлы:
- "LR01_environment.ngscript" ("LR01_environment.ngscript" -> "LR01"),
- "test.jl" ("test.js" -> "LR01").

## Результаты команд

Результат команды "VERSION": v"1.12.4"

Результат команды "versioninfo()":

Julia Version 1.12.4
Commit 01a2eadb047 (2026-01-06 16:56 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 112 × Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, skylake-avx512)
  GC: Built with stock GC
Threads: 5 default, 3 interactive, 5 GC (on 112 virtual cores)
Environment:
  JULIA_LSP_STORE_PATH = /user/lsp_store
  JULIA_CPU_TARGET = generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1)
  JULIA_LSP_PORT = 3751
  JULIA_NUM_PRECOMPILE_TASKS = 8
  JULIA_PKG_PRECOMPILE_AUTO = 0
  JULIA_DEPOT_PATH = /user/.packages:/opt/depot/core:/opt/depot/mtl:/opt/depot/demos:/opt/depot/stdlib
  JULIA_PATH = /usr/local/julia
  LD_LIBRARY_PATH = /usr/local/nvidia/lib:/usr/local/nvidia/lib64
  JULIA_PROJECT = /user/.project/
  JULIA_DA_STORE_PATH = /user/da_store
  JULIA_PKG_SERVER = http://pkgserver.pkgserver.svc.cluster.local
  JULIA_NUM_THREADS = 7,1
  JULIA_DA_PORT = 3752
  JULIA_DEBUG = BackendLib,AbstractAPI

## Результат объявления переменной

В файле "test.jl" объявлена переменная seed со значением 101

вход: seed = 101

вывод: include("/user/newFile_0.jl")
101

В таблице переменных появилось 2 переменные: "ans" и "seed" со значением 101.

## Результат очистки таблицы переменных

После выполнении действия "Очистить все переменные" из таблицы переменных удалилась переменная "seed".

После выполнения действия "Перезапуск ядра" из таблицы переменных удалилась переменная "ans".

Вывод: Состояние среды в Engee зависит от порядка выполнения ячеек. Переменная seed появляется в рабочей области только после выполнения ячейки с её объявлением. Команда «Очистить все переменные» удаляет переменные, но не сбрасывает ядро. Команда «Перезапуск ядра» полностью сбрасывает среду - после неё переменные не существуют, пока ячейка не будет выполнена снова. Поэтому в паспорт версии записывается порядок запуска ячеек, а не только их содержимое.