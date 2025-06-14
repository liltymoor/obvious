"""Golden тесты транслятора и машины.
Конфигурационнфе файлы: "golden/*.yml"
"""

import ast
import contextlib
import io
import logging
import os
import tempfile

import pytest

import builder
import runner

MAX_LOG = 4000


def parse_yaml_input(s: str) -> list[tuple[int, str]]:
    data = ast.literal_eval(s)

    return [(int(num), str(char)) for num, char in data]


def get_last_n_lines(text: str, n: int) -> str:
    lines = text.splitlines()
    last_n_lines = lines[-n:] if len(lines) > n else lines
    return "\n".join(last_n_lines)


@pytest.mark.golden_test("../golden/*.yml")
def test_translator_and_machine(golden, caplog):
    """
    Вход:
    - `in_source` -- исходный код
    - `in_stdin` -- данные на ввод процессора для симуляции
    Выход:

    - `out_hex` -- аннотированный машинный код
    - `out_binary` -- бинарный файл в base64
    - `out_stdout` -- стандартный вывод транслятора и симулятора
    - `out_log` -- журнал программы
    """
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s:root:%(message)s")
    caplog.set_level(logging.DEBUG)

    with tempfile.TemporaryDirectory() as tmpdirname:
        source = os.path.join(tmpdirname, "test_src.obv")
        target = os.path.join(tmpdirname, f"target_{golden.name}")
        target_hex = os.path.join(tmpdirname, f"target_{golden.name}.hex")
        journal = os.path.join(tmpdirname, "journal.txt")

        # Записываем входные данные в файлы. Данные берутся из теста.
        with open(source, "w", encoding="utf-8") as file:
            file.write(golden["in_source"])
        input_stream = parse_yaml_input(golden["in_stdin"])

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            builder.build(source, target)
            print("============================================================")
            runner.run(target + ".bin", input_stream, verbose=True)
        with open(target + ".bin", "rb") as file:
            code_bin = file.read()
        with open(target_hex, encoding="utf-8") as file:
            code_hex = file.read()
        with open(journal) as file:
            journal_text = file.read()

        assert code_bin == golden.out["out_code_bin"]
        assert code_hex == golden.out["out_code_hex"]
        assert stdout.getvalue() == golden.out["out_stdout"]
        assert get_last_n_lines(journal_text, 10) == golden.out["out_log"]
