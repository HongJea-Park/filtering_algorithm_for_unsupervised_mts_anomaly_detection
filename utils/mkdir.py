import os


def mkdir(base_dir):

    base_split = base_dir.replace('../', '').split('/')

    folder = base_split[:-1]
    file = base_split[-1]

    dir_ = '.'
    for fold in folder:

        dir_ += f'/{fold}'

        if not os.path.isdir(dir_):
            os.mkdir(dir_)

    return f'{dir_}/{file}'
