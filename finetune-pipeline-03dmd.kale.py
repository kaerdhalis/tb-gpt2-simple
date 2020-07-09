import kfp.dsl as dsl
import kfp.components as comp
from collections import OrderedDict
from kubernetes import client as k8s_client


def load_data_helper(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_data_helper",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # helper to use code samples in zip file
    def process_zip(name, regs, postfix, data_dir, min_length, max_length, preserve_form, num_samples):
        with open(os.path.join(output_dir, name + postfix + '.txt'), 'w+') as fh:
            with zipfile.ZipFile(os.path.join(data_dir, name + '.zip'), 'r') as z:
                cnt = 0
                for entry in z.namelist():
                    text = z.read(entry).decode('utf-8')
                    for reg, sub in regs.items():
                        text = re.sub(reg, sub, text, flags=re.DOTALL)
                    if len(text) > min_length and len(text) <= max_length:
                        sample = text.strip() + "\n"
                        if preserve_form == 'true':
                            sample += "\n\n"
                        fh.write(sample)
                        cnt += 1
                    if cnt >= num_samples:
                        break

    # command line arguments parser
    data_type = 'all'
    data_dir = './datasets/'
    output_dir = './data'
    short_filename = 'true'
    postfix = ''
    num_samples = 1000
    max_length = 2000
    min_length = 10
    preserve_lines = 'true'
    preserve_form = 'false'

    # form requires newlines to be preserved
    if preserve_form == 'true':
        preserve_lines = 'true'

    # collapsing sample into one line requires form not to be preserved
    if preserve_lines == 'false':
        preserve_form = 'false'

    # set postfix for output files if short-filename is false
    if postfix != '':
        postfix = '_' + postfix
    if short_filename == 'false':
        postfix += f'_n{num_samples}_min{min_length}_max{max_length}'
        if preserve_lines == 'false':
            postfix += '_nolines'
        else:
            postfix += '_lines'
        if preserve_form == 'false':
            postfix += '_noform'
        else:
            postfix += '_form'

    # -----------------------DATA SAVING START---------------------------------
    if "min_length" in locals():
        _kale_resource_save(min_length, os.path.join(
            _kale_data_directory, "min_length"))
    else:
        print("_kale_resource_save: `min_length` not found.")
    if "num_samples" in locals():
        _kale_resource_save(num_samples, os.path.join(
            _kale_data_directory, "num_samples"))
    else:
        print("_kale_resource_save: `num_samples` not found.")
    if "postfix" in locals():
        _kale_resource_save(postfix, os.path.join(
            _kale_data_directory, "postfix"))
    else:
        print("_kale_resource_save: `postfix` not found.")
    if "data_dir" in locals():
        _kale_resource_save(data_dir, os.path.join(
            _kale_data_directory, "data_dir"))
    else:
        print("_kale_resource_save: `data_dir` not found.")
    if "preserve_lines" in locals():
        _kale_resource_save(preserve_lines, os.path.join(
            _kale_data_directory, "preserve_lines"))
    else:
        print("_kale_resource_save: `preserve_lines` not found.")
    if "max_length" in locals():
        _kale_resource_save(max_length, os.path.join(
            _kale_data_directory, "max_length"))
    else:
        print("_kale_resource_save: `max_length` not found.")
    if "data_type" in locals():
        _kale_resource_save(data_type, os.path.join(
            _kale_data_directory, "data_type"))
    else:
        print("_kale_resource_save: `data_type` not found.")
    if "process_zip" in locals():
        _kale_resource_save(process_zip, os.path.join(
            _kale_data_directory, "process_zip"))
    else:
        print("_kale_resource_save: `process_zip` not found.")
    if "output_dir" in locals():
        _kale_resource_save(output_dir, os.path.join(
            _kale_data_directory, "output_dir"))
    else:
        print("_kale_resource_save: `output_dir` not found.")
    if "preserve_form" in locals():
        _kale_resource_save(preserve_form, os.path.join(
            _kale_data_directory, "preserve_form"))
    else:
        print("_kale_resource_save: `preserve_form` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def load_html_data(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_html_data",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "data_type" not in _kale_directory_file_names:
        raise ValueError("data_type" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_type"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_type" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_type = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "num_samples" not in _kale_directory_file_names:
        raise ValueError("num_samples" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "num_samples"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "num_samples" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    num_samples = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "postfix" not in _kale_directory_file_names:
        raise ValueError("postfix" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "postfix"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "postfix" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    postfix = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "output_dir" not in _kale_directory_file_names:
        raise ValueError("output_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "output_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "output_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    output_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_dir" not in _kale_directory_file_names:
        raise ValueError("data_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # dataset from: https://www.kaggle.com/zavadskyy/lots-of-code, https://gist.github.com/VladislavZavadskyy/e31ab07b03a5c22b11982c49669a400b
    if data_type in ['all', 'html']:  # parse html
        print('prepare html data set...')
        with open(os.path.join(output_dir, 'html' + postfix + '.txt'), 'w+') as fh:
            with open(os.path.join(data_dir, 'html-dataset.txt')) as fp:
                data = fp.read()
                data = data.replace('<!DOCTYPE html>', '\n<!DOCTYPE html>')
                lines = data.split('\n')
                cnt = 0
                sample = ""
                for line in lines:
                    if line == "":
                        continue
                    if sample != "" and line.startswith('<!DOCTYPE html>'):
                        fh.write(sample.strip() + "\n")
                        sample = ""
                        cnt += 1
                    if cnt >= num_samples:
                        break
                    line = re.sub(r'\s+', ' ', line)
                    sample += line.strip() + " "
        print('preparing html data set done.')


def load_json_data(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_json_data",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "min_length" not in _kale_directory_file_names:
        raise ValueError("min_length" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "min_length"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "min_length" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    min_length = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "num_samples" not in _kale_directory_file_names:
        raise ValueError("num_samples" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "num_samples"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "num_samples" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    num_samples = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "postfix" not in _kale_directory_file_names:
        raise ValueError("postfix" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "postfix"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "postfix" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    postfix = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_dir" not in _kale_directory_file_names:
        raise ValueError("data_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_lines" not in _kale_directory_file_names:
        raise ValueError("preserve_lines" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_lines"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_lines" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_lines = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "max_length" not in _kale_directory_file_names:
        raise ValueError("max_length" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "max_length"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "max_length" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    max_length = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_type" not in _kale_directory_file_names:
        raise ValueError("data_type" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_type"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_type" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_type = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "process_zip" not in _kale_directory_file_names:
        raise ValueError("process_zip" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "process_zip"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "process_zip" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    process_zip = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_form" not in _kale_directory_file_names:
        raise ValueError("preserve_form" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_form"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_form" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_form = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # dataset from: json files collected from standard angular app
    if data_type in ['all', 'json']:  # parse json files
        print('prepare json data set...')
        regexes = {}
        if preserve_lines == 'false':
            regexes[r'\s+'] = ' '
        process_zip('json', regexes, postfix, data_dir, min_length,
                    max_length, preserve_form, num_samples)
        print('preparing json data set done.')


def load_typescript_data(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_typescript_data",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "min_length" not in _kale_directory_file_names:
        raise ValueError("min_length" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "min_length"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "min_length" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    min_length = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "num_samples" not in _kale_directory_file_names:
        raise ValueError("num_samples" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "num_samples"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "num_samples" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    num_samples = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "postfix" not in _kale_directory_file_names:
        raise ValueError("postfix" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "postfix"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "postfix" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    postfix = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_dir" not in _kale_directory_file_names:
        raise ValueError("data_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_lines" not in _kale_directory_file_names:
        raise ValueError("preserve_lines" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_lines"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_lines" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_lines = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "max_length" not in _kale_directory_file_names:
        raise ValueError("max_length" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "max_length"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "max_length" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    max_length = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_type" not in _kale_directory_file_names:
        raise ValueError("data_type" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_type"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_type" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_type = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "process_zip" not in _kale_directory_file_names:
        raise ValueError("process_zip" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "process_zip"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "process_zip" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    process_zip = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_form" not in _kale_directory_file_names:
        raise ValueError("preserve_form" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_form"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_form" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_form = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # dataset from: typescript files collected from standard angular app
    if data_type in ['all', 'typescript']:  # parse typescript files
        print('prepare typescript data set...')
        regexes = {}
        if preserve_form == 'false':
            regexes[r'(//[^\n]*)?\n|/\*.*?\*/'] = '\n'
            regexes[r'\n\s*\n'] = '\n'
        if preserve_lines == 'false':
            regexes[r'\s+'] = ' '
        process_zip('typescript', regexes, postfix, data_dir,
                    min_length, max_length, preserve_form, num_samples)
        print('preparing typescript data set done.')


def load_javascript_data(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_javascript_data",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "min_length" not in _kale_directory_file_names:
        raise ValueError("min_length" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "min_length"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "min_length" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    min_length = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "num_samples" not in _kale_directory_file_names:
        raise ValueError("num_samples" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "num_samples"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "num_samples" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    num_samples = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "postfix" not in _kale_directory_file_names:
        raise ValueError("postfix" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "postfix"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "postfix" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    postfix = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_dir" not in _kale_directory_file_names:
        raise ValueError("data_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_lines" not in _kale_directory_file_names:
        raise ValueError("preserve_lines" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_lines"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_lines" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_lines = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "max_length" not in _kale_directory_file_names:
        raise ValueError("max_length" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "max_length"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "max_length" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    max_length = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_type" not in _kale_directory_file_names:
        raise ValueError("data_type" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_type"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_type" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_type = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "process_zip" not in _kale_directory_file_names:
        raise ValueError("process_zip" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "process_zip"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "process_zip" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    process_zip = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_form" not in _kale_directory_file_names:
        raise ValueError("preserve_form" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_form"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_form" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_form = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # dataset from: javascript files from https://www.sri.inf.ethz.ch/js150
    if data_type in ['all', 'javascript']:  # parse javascript files
        print('prepare javascript data set...')
        regexes = {}
        if preserve_form == 'false':
            regexes[r'(//[^\n]*)?\n|/\*.*?\*/'] = '\n'
            regexes[r'\n\s*\n'] = '\n'
        if preserve_lines == 'false':
            regexes[r'\s+'] = ' '
        process_zip('javascript', regexes, postfix, data_dir,
                    min_length, max_length, preserve_form, num_samples)
        print('preparing javascript data set done.')


def load_shakespeare_data(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_shakespeare_data",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "data_type" not in _kale_directory_file_names:
        raise ValueError("data_type" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_type"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_type" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_type = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "num_samples" not in _kale_directory_file_names:
        raise ValueError("num_samples" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "num_samples"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "num_samples" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    num_samples = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "postfix" not in _kale_directory_file_names:
        raise ValueError("postfix" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "postfix"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "postfix" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    postfix = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "output_dir" not in _kale_directory_file_names:
        raise ValueError("output_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "output_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "output_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    output_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_dir" not in _kale_directory_file_names:
        raise ValueError("data_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_lines" not in _kale_directory_file_names:
        raise ValueError("preserve_lines" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_lines"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_lines" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_lines = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # dataset from: https://www.kaggle.com/kingburrito666/shakespeare-plays
    if data_type in ['all', 'shakespeare']:  # parse shakespeare plays
        print('prepare shakespeare data set...')
        df = pd.read_csv(os.path.join(data_dir, 'shakespeare_data.csv'))
        if preserve_lines == 'false':
            df = df[df.Player != ''].groupby(
                ['Play', 'PlayerLinenumber'], as_index=False).agg(' '.join)
        df.sample(num_samples).PlayerLine.to_csv(os.path.join(output_dir, 'shakespeare' + postfix +
                                                              '.txt'), index=False, header=False, quoting=csv.QUOTE_NONE, escapechar="\\", sep="\\")
        print('preparing shakespeare data set done.')


def load_music_data(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_music_data",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "data_type" not in _kale_directory_file_names:
        raise ValueError("data_type" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_type"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_type" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_type = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "num_samples" not in _kale_directory_file_names:
        raise ValueError("num_samples" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "num_samples"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "num_samples" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    num_samples = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "postfix" not in _kale_directory_file_names:
        raise ValueError("postfix" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "postfix"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "postfix" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    postfix = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "output_dir" not in _kale_directory_file_names:
        raise ValueError("output_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "output_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "output_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    output_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_dir" not in _kale_directory_file_names:
        raise ValueError("data_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_lines" not in _kale_directory_file_names:
        raise ValueError("preserve_lines" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_lines"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_lines" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_lines = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # dataset from: https://www.kaggle.com/raj5287/abc-notation-of-tunes/version/3
    if data_type in ['all', 'music']:  # parse abc songs
        print('prepare music data set...')
        with open(os.path.join(output_dir, 'music' + postfix + '.txt'), 'w+') as fh:
            with open(os.path.join(data_dir, 'abc_notation_songs.txt')) as fp:
                line = fp.readline()
                cnt = 0
                song = ""
                while line and cnt < num_samples:
                    if len(line) < 2 or line[1:2] == ':':
                        if song != "":
                            fh.write(song + "\n")
                            cnt += 1
                            song = ""
                    elif preserve_lines == 'false':
                        song += " " + line.strip()
                    else:
                        fh.write(line.strip() + "\n")
                    line = fp.readline()
        print('preparing music data set done.')


def finetune_model(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "finetune_model",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "stpes" not in _kale_directory_file_names:
        raise ValueError("stpes" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "stpes"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "stpes" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    stpes = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    def start_session(sess):
        try:
            gpt2.reset_session(sess)
        except:
            pass
        return gpt2.start_tf_sess()

    def fine_tune(sess, run_name, data_path, steps, model_name='124M'):
        print(
            f'Run fine-tuning for run {run_name} using GPT2 model {model_name}...')
        if not os.path.isdir(os.path.join("models", model_name)):
            log.info(f"Downloading {model_name} model...")
            gpt2.download_gpt2(model_name=model_name)
        sess = start_session(sess)
        gpt2.finetune(sess=sess, dataset=data_path, checkpoint_dir='runs', model_name=model_name,
                      run_name=run_name, steps=steps, sample_every=10, save_every=10)

    # run_name='run1'
    #data_path ='data/tweets.txt'
    sess = None
    fine_tune(sess, run_name, data_path, stpes)

    # -----------------------DATA SAVING START---------------------------------
    if "start_session" in locals():
        _kale_resource_save(start_session, os.path.join(
            _kale_data_directory, "start_session"))
    else:
        print("_kale_resource_save: `start_session` not found.")
    if "sess" in locals():
        _kale_resource_save(sess, os.path.join(_kale_data_directory, "sess"))
    else:
        print("_kale_resource_save: `sess` not found.")
    # -----------------------DATA SAVING END-----------------------------------


def generate_text(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "generate_text",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "start_session" not in _kale_directory_file_names:
        raise ValueError("start_session" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "start_session"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "start_session" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    start_session = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "sess" not in _kale_directory_file_names:
        raise ValueError("sess" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "sess"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "sess" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    sess = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    def generate(sess, run_name, length, temperature, top_k):

        message = "Social justice warior"
        text = gpt2.generate(sess=sess, checkpoint_dir='runs', run_name=run_name, prefix=message,
                             length=length, temperature=temperature, top_k=top_k, return_as_list=True)
        print(text[0])

    # length = 800 # { min:0, max:1000, step:5}
    # temperature = 0.7 # { min:0, max:2, step:0.1}
    #top_k = 0
    sess = start_session(sess)
    gpt2.load_gpt2(sess, checkpoint_dir='runs', run_name=run_name)
    generate(sess, run_name, length, temperature, top_k)


def load_chess_data(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_chess_data",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "data_type" not in _kale_directory_file_names:
        raise ValueError("data_type" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_type"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_type" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_type = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "num_samples" not in _kale_directory_file_names:
        raise ValueError("num_samples" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "num_samples"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "num_samples" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    num_samples = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "postfix" not in _kale_directory_file_names:
        raise ValueError("postfix" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "postfix"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "postfix" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    postfix = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "output_dir" not in _kale_directory_file_names:
        raise ValueError("output_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "output_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "output_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    output_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_dir" not in _kale_directory_file_names:
        raise ValueError("data_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # dataset from: https://www.ficsgames.org/download.html | year: 2019, month: whole year, type: Standard (average rating > 2000)
    if data_type in ['all', 'chess']:  # parse chess games
        print('prepare chess data set...')
        with open(os.path.join(output_dir, 'chess' + postfix + '.txt'), 'w+') as fh:
            with open(os.path.join(data_dir, 'ficsgamesdb_2019_standard2000_nomovetimes_110541.pgn')) as fp:
                line = fp.readline()
                cnt = 0
                while line and cnt < num_samples:
                    if line.startswith('1.'):
                        fh.write(line)
                        cnt += 1
                    line = fp.readline()
        print('preparing chess data set done.')


def load_tweet_data(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "load_tweet_data",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")

    # -----------------------DATA LOADING START--------------------------------
    _kale_directory_file_names = [
        os.path.splitext(f)[0]
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f))
    ]

    if "min_length" not in _kale_directory_file_names:
        raise ValueError("min_length" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "min_length"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "min_length" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    min_length = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "num_samples" not in _kale_directory_file_names:
        raise ValueError("num_samples" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "num_samples"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "num_samples" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    num_samples = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "postfix" not in _kale_directory_file_names:
        raise ValueError("postfix" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "postfix"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "postfix" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    postfix = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_dir" not in _kale_directory_file_names:
        raise ValueError("data_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_lines" not in _kale_directory_file_names:
        raise ValueError("preserve_lines" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_lines"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_lines" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_lines = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "data_type" not in _kale_directory_file_names:
        raise ValueError("data_type" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "data_type"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "data_type" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    data_type = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "output_dir" not in _kale_directory_file_names:
        raise ValueError("output_dir" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "output_dir"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "output_dir" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    output_dir = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))

    if "preserve_form" not in _kale_directory_file_names:
        raise ValueError("preserve_form" + " does not exists in directory")

    _kale_load_file_name = [
        f
        for f in os.listdir(_kale_data_directory)
        if os.path.isfile(os.path.join(_kale_data_directory, f)) and
        os.path.splitext(f)[0] == "preserve_form"
    ]
    if len(_kale_load_file_name) > 1:
        raise ValueError("Found multiple files with name " +
                         "preserve_form" + ": " + str(_kale_load_file_name))
    _kale_load_file_name = _kale_load_file_name[0]
    preserve_form = _kale_resource_load(os.path.join(
        _kale_data_directory, _kale_load_file_name))
    # -----------------------DATA LOADING END----------------------------------

    import gpt_2_simple as gpt2
    import json
    import os
    import sys
    import numpy as np
    import argparse
    import requests
    import glob
    import pickle
    import pandas as pd
    import re
    import unicodedata
    import csv
    import zipfile
    import logging
    logging.basicConfig(
        level=logging.DEBUG, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
    log = logging.getLogger(__name__)

    # dataset from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi%3A10.7910%2FDVN%2FKJEBIL
    if data_type in ['all', 'tweets']:  # parse trump tweets
        print('prepare tweet data set...')
        df1 = pd.read_json(os.path.join(
            data_dir, 'realdonaldtrump-1.ndjson'), lines=True)
        df2 = pd.read_json(os.path.join(
            data_dir, 'realdonaldtrump-2.ndjson'), lines=True)
        df = pd.concat([df1, df2], sort=True)
        if preserve_lines == 'false':
            df.text = df.text.str.replace("\n", " ")
        if preserve_form == 'false':
            df.text = df.text.str.replace(r"https?://[^\s]+", "")
        df['length'] = df.text.apply(len)
        filter = (df.text > '2017') & (df.text.str.startswith(
            'RT') == False) & (df.length > min_length)
        df = df[filter]
        df.sample(num_samples).text.to_csv(os.path.join(output_dir, 'tweets' + postfix + '.txt'),
                                           index=False, header=False, quoting=csv.QUOTE_NONE, escapechar="\\", sep="\\")
        print('preparing tweet data set done.')


def final_auto_snapshot(run_name: str, data_path: str, steps: int, length: int, temperature: float, top_k: int, rok_workspace_tb_finetuning_lmy6scgb1_url: str):

    import os
    import shutil
    from kale.utils import pod_utils
    from kale.marshal import resource_save as _kale_resource_save
    from kale.marshal import resource_load as _kale_resource_load

    _kale_data_directory = "/home/jovyan/tb-gpt2-simple/.finetune.ipynb.kale.marshal.dir"

    if not os.path.isdir(_kale_data_directory):
        os.makedirs(_kale_data_directory, exist_ok=True)

    pod_utils.snapshot_pipeline_step("finetune-pipeline-03dmd",
                                     "final_auto_snapshot",
                                     "/home/jovyan/tb-gpt2-simple/finetune.ipynb")


load_data_helper_op = comp.func_to_container_op(
    load_data_helper, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


load_html_data_op = comp.func_to_container_op(
    load_html_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


load_json_data_op = comp.func_to_container_op(
    load_json_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


load_typescript_data_op = comp.func_to_container_op(
    load_typescript_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


load_javascript_data_op = comp.func_to_container_op(
    load_javascript_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


load_shakespeare_data_op = comp.func_to_container_op(
    load_shakespeare_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


load_music_data_op = comp.func_to_container_op(
    load_music_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


finetune_model_op = comp.func_to_container_op(
    finetune_model, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


generate_text_op = comp.func_to_container_op(
    generate_text, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


load_chess_data_op = comp.func_to_container_op(
    load_chess_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


load_tweet_data_op = comp.func_to_container_op(
    load_tweet_data, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


final_auto_snapshot_op = comp.func_to_container_op(
    final_auto_snapshot, base_image='gcr.io/arrikto-public/tensorflow-1.15.2-notebook-cpu:1.0.0.arr1')


@dsl.pipeline(
    name='finetune-pipeline-03dmd',
    description='finetuning basic'
)
def auto_generated_pipeline(run_name='run1', data_path='data/tweets.txt', steps='1', length='800', temperature='0.7', top_k='0', rok_workspace_tb_finetuning_lmy6scgb1_url='http://rok.rok.svc.cluster.local/swift/v1/c1152cd8-6699-4f0f-a65a-151f12ef6e8f/notebooks/tb-finetuning-0_workspace-tb-finetuning-lmy6scgb1?version=abf56ee1-8628-44aa-a743-1ade5bfecc93'):
    pvolumes_dict = OrderedDict()

    annotations = {'rok/origin': 'http://rok.rok.svc.cluster.local/swift/v1/c1152cd8-6699-4f0f-a65a-151f12ef6e8f/notebooks/tb-finetuning-0_workspace-tb-finetuning-lmy6scgb1?version=abf56ee1-8628-44aa-a743-1ade5bfecc93'}

    annotations['rok/origin'] = rok_workspace_tb_finetuning_lmy6scgb1_url

    vop1 = dsl.VolumeOp(
        name='create-volume-1',
        resource_name='workspace-tb-finetuning-lmy6scgb1',
        annotations=annotations,
        size='5Gi'
    )
    volume = vop1.volume

    pvolumes_dict['/home/jovyan'] = volume

    load_data_helper_task = load_data_helper_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after()
    load_data_helper_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_data_helper_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_data_helper_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    load_html_data_task = load_html_data_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_helper_task)
    load_html_data_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_html_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_html_data_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    load_json_data_task = load_json_data_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_helper_task)
    load_json_data_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_json_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_json_data_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    load_typescript_data_task = load_typescript_data_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_helper_task)
    load_typescript_data_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_typescript_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_typescript_data_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)

    load_javascript_data_task = load_javascript_data_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_helper_task)
    load_javascript_data_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_javascript_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_javascript_data_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)

    load_shakespeare_data_task = load_shakespeare_data_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_helper_task)
    load_shakespeare_data_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_shakespeare_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_shakespeare_data_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)

    load_music_data_task = load_music_data_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_helper_task)
    load_music_data_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_music_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_music_data_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    finetune_model_task = finetune_model_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_music_data_task)
    finetune_model_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    finetune_model_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    finetune_model_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    generate_text_task = generate_text_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(finetune_model_task)
    generate_text_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    generate_text_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    generate_text_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    load_chess_data_task = load_chess_data_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_helper_task)
    load_chess_data_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_chess_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_chess_data_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    load_tweet_data_task = load_tweet_data_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_data_helper_task)
    load_tweet_data_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    load_tweet_data_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    load_tweet_data_task.output_artifact_paths.update(mlpipeline_ui_metadata)

    final_auto_snapshot_task = final_auto_snapshot_op(run_name, data_path, steps, length, temperature, top_k, rok_workspace_tb_finetuning_lmy6scgb1_url)\
        .add_pvolumes(pvolumes_dict)\
        .after(load_tweet_data_task, load_chess_data_task, load_shakespeare_data_task, load_javascript_data_task, load_typescript_data_task, load_json_data_task, load_html_data_task, generate_text_task)
    final_auto_snapshot_task.container.working_dir = "/home/jovyan/tb-gpt2-simple"
    final_auto_snapshot_task.container.set_security_context(
        k8s_client.V1SecurityContext(run_as_user=0))
    mlpipeline_ui_metadata = {
        'mlpipeline-ui-metadata': '/mlpipeline-ui-metadata.json'}
    final_auto_snapshot_task.output_artifact_paths.update(
        mlpipeline_ui_metadata)


if __name__ == "__main__":
    pipeline_func = auto_generated_pipeline
    pipeline_filename = pipeline_func.__name__ + '.pipeline.tar.gz'
    import kfp.compiler as compiler
    compiler.Compiler().compile(pipeline_func, pipeline_filename)

    # Get or create an experiment and submit a pipeline run
    import kfp
    client = kfp.Client()
    experiment = client.create_experiment('Gpt2-simple')

    # Submit a pipeline run
    run_name = 'finetune-pipeline-03dmd_run'
    run_result = client.run_pipeline(
        experiment.id, run_name, pipeline_filename, {})
