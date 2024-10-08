import os

path = './data/SceneFlow/'
mode = 'test' # train, test

left_imgs = list()
right_imgs = list()
disp_imgs = list()


def add_driving():
    root_driving = os.path.join(path, 'Driving')

    img_fmt = '{:04}.png'
    disp_fmt = '{:04}.pfm'
    driving_left_imgs = list()
    driving_right_imgs = list()
    driving_disp_imgs = list()

    driving_path = os.path.join(root_driving, 'frames_finalpass')
    disp_driving_path = os.path.join(root_driving, 'disparity')

    driving_towards = list()
    driving_towards.append(os.path.join(driving_path, '15mm_focallength/scene_backwards'))
    driving_towards.append(os.path.join(driving_path, '15mm_focallength/scene_forwards'))
    driving_towards.append(os.path.join(driving_path, '35mm_focallength/scene_backwards'))
    driving_towards.append(os.path.join(driving_path, '35mm_focallength/scene_forwards'))


    for towards in driving_towards:
        for i in range(1, 301):
            driving_left_imgs.append(os.path.join(towards, 'fast/left', img_fmt.format(i)))
            driving_right_imgs.append(os.path.join(towards, 'fast/right', img_fmt.format(i)))

        for j in range(1, 801):
            driving_left_imgs.append(os.path.join(towards, 'slow/left', img_fmt.format(j)))
            driving_right_imgs.append(os.path.join(towards, 'slow/right', img_fmt.format(j)))

    disp_driving_towards = list()
    disp_driving_towards.append(os.path.join(disp_driving_path, '15mm_focallength/scene_backwards'))
    disp_driving_towards.append(os.path.join(disp_driving_path, '15mm_focallength/scene_forwards'))
    disp_driving_towards.append(os.path.join(disp_driving_path, '35mm_focallength/scene_backwards'))
    disp_driving_towards.append(os.path.join(disp_driving_path, '35mm_focallength/scene_forwards'))


    for i in disp_driving_towards:
        for j in range(1, 301):
            driving_disp_imgs.append(os.path.join(i, 'fast/left', disp_fmt.format(j)))

        for k in range(1, 801):
            driving_disp_imgs.append(os.path.join(i, 'slow/left', disp_fmt.format(k)))

    return driving_left_imgs, driving_right_imgs, driving_disp_imgs


def add_monkaa():
    root_monkaa = os.path.join(path, 'Monkaa')

    monkaa_left_imgs = list()
    monkaa_right_imgs = list()
    monkaa_disp_imgs = list()

    monkaa_path = os.path.join(root_monkaa, 'frames_finalpass')
    disp_monkaa_path = os.path.join(root_monkaa, 'disparity')

    monkaa_scenes = sorted(os.listdir(monkaa_path))


    for scene in monkaa_scenes:
        scene_left = os.path.join(monkaa_path, scene, 'left')
        monkaa_left_imgs.extend(os.path.join(scene_left, img) for img in sorted(os.listdir(scene_left)))

        scene_right = os.path.join(monkaa_path, scene, 'right')
        monkaa_right_imgs.extend(os.path.join(scene_right, img) for img in sorted(os.listdir(scene_right)))

        disp_scene = os.path.join(disp_monkaa_path, scene, 'left')
        monkaa_disp_imgs.extend(os.path.join(disp_scene, img) for img in sorted(os.listdir(disp_scene)))

    return monkaa_left_imgs, monkaa_right_imgs, monkaa_disp_imgs


def add_flyingthings3d():
    root_flyingthings3d = os.path.join(path, 'FlyingThings3D')

    flyingthings3d_left_imgs = list()
    flyingthings3d_right_imgs = list()
    flyingthings3d_disp_imgs = list()

    flyingthings3d_path = os.path.join(root_flyingthings3d, 'frames_finalpass')
    disp_flyingthings3d_path = os.path.join(root_flyingthings3d, 'disparity')

    for mode in ['TRAIN']: # ['TEST', 'TRAIN']
        for abc in ['A', 'B', 'C']:
            img_path_abc = os.path.join(flyingthings3d_path, mode, abc)
            disp_path_abc = os.path.join(disp_flyingthings3d_path, mode, abc)
            flyingthings3d_folders = sorted(os.listdir(img_path_abc))

            for folder in flyingthings3d_folders:
                folder_left = os.path.join(img_path_abc, folder, 'left')
                flyingthings3d_left_imgs.extend(os.path.join(folder_left, img) for img in sorted(os.listdir(folder_left)))

                folder_right = os.path.join(img_path_abc, folder, 'right')
                flyingthings3d_right_imgs.extend(os.path.join(folder_right, img) for img in sorted(os.listdir(folder_right)))

                folder_disp = os.path.join(disp_path_abc, folder, 'left')
                flyingthings3d_disp_imgs.extend(os.path.join(folder_disp, img) for img in sorted(os.listdir(folder_disp)))

    return flyingthings3d_left_imgs, flyingthings3d_right_imgs, flyingthings3d_disp_imgs

if mode == 'train':
    driving_left_imgs, driving_right_imgs, driving_disp_imgs = add_driving()
    monkaa_left_imgs, monkaa_right_imgs, monkaa_disp_imgs = add_monkaa()
    flyingthings3d_left_imgs, flyingthings3d_right_imgs, flyingthings3d_disp_imgs = add_flyingthings3d()

    left_imgs = driving_left_imgs + monkaa_left_imgs + flyingthings3d_left_imgs
    right_imgs = driving_right_imgs + monkaa_right_imgs + flyingthings3d_right_imgs
    disp_imgs = driving_disp_imgs + monkaa_disp_imgs + flyingthings3d_disp_imgs
elif mode == 'test':
    root = os.path.join(path, 'FlyingThings3D')
    img_path = os.path.join(root, 'frames_finalpass/TEST')
    disp_path = os.path.join(root, 'disparity/TEST')

    for abc in ['A', 'B', 'C']:
        img_path_abc = os.path.join(img_path, abc)
        disp_path_abc = os.path.join(disp_path, abc)
        img_folders = sorted(os.listdir(img_path_abc))

        for folder in img_folders:
            folder_left = os.path.join(img_path_abc, folder, 'left')
            left_imgs.extend(os.path.join(folder_left, img) for img in sorted(os.listdir(folder_left)))

            folder_right = os.path.join(img_path_abc, folder, 'right')
            right_imgs.extend(os.path.join(folder_right, img) for img in sorted(os.listdir(folder_right)))

            disp_folder = os.path.join(disp_path_abc, folder, 'left')
            disp_imgs.extend(os.path.join(disp_folder, disp) for disp in sorted(os.listdir(disp_folder)))

if mode == 'train': f = open('sceneflow_train0.txt', 'w')
else: f = open('sceneflow_test0.txt', 'w')
for index, (img_l, img_r, disp) in enumerate(zip(left_imgs, right_imgs, disp_imgs)):
    line = img_l.replace(path, '') + ' ' + img_r.replace(path, '') + ' ' + disp.replace(path, '')
    f.write(line + '\n')
    print('index', index)

f.close()
