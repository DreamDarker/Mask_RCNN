dataset_root_path = "C:/Users/12084/Desktop/Proj/data/7-11/"


def get_root_path():
    return dataset_root_path


def get_img_folder():
    return dataset_root_path + "img-resized"


def get_mask_folder():
    return dataset_root_path + "mask8"


def get_yaml_path(filestr):
    return dataset_root_path + "dataset/" + filestr + "_json/info.yaml"


def get_img_path(filestr):
    return dataset_root_path + "dataset/" + filestr + "_json/img.png"
    # return dataset_root_path + "img G" + "/" + filestr + ".png"


def get_mask_path(filestr):
    # return dataset_root_path + "dataset/" + filestr + "_json/label.png"
    return dataset_root_path + "mask8" + "/" + filestr + ".png"


def main():
    print(get_img_path('1'))
    print(get_mask_path('2'))
    print(get_yaml_path('3'))


if __name__ == '__main__':
    main()
