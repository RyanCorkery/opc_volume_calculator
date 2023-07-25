import cv2
import numpy as np
import pyvista as pv
from pathlib import Path
import PVGeo
import scipy.spatial as ss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# steps
# loop through images
# mask images on red filter
# erode and dilate
# find contours
# save contours to point cloud list
# convert cloud point to mesh
# plot mesh
# calculate volume

# x=0.07um/pixel, y=0.07um/pixel, z=1.0um/pixel


class OPCDetector:
    min_h = 0
    min_s = 15
    min_v = 0

    max_h = 12
    max_s = 255
    max_v = 255

    x_um_per_px = 0.07
    y_um_per_px = 0.07
    z_um_per_layer = 1.0

    roi = ...

    def __init__(self):
        self.img_paths = ...

    # def get_image(self, path: str):
    #     pass

    @staticmethod
    def doNothing(*args):
        pass

    def find_hsv_values(self, img):
        cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

        cv2.createTrackbar('min_H', 'Track Bars', 0, 255, self.doNothing)
        cv2.createTrackbar('min_S', 'Track Bars', 0, 255, self.doNothing)
        cv2.createTrackbar('min_V', 'Track Bars', 0, 255, self.doNothing)

        cv2.createTrackbar('max_H', 'Track Bars', 0, 255, self.doNothing)
        cv2.createTrackbar('max_S', 'Track Bars', 0, 255, self.doNothing)
        cv2.createTrackbar('max_V', 'Track Bars', 0, 255, self.doNothing)

        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow('hsv image', img_hsv)
        cv2.imshow('image', img)

        while True:
            min_h = cv2.getTrackbarPos('min_H', 'Track Bars')
            min_s = cv2.getTrackbarPos('min_S', 'Track Bars')
            min_v = cv2.getTrackbarPos('min_V', 'Track Bars')

            max_h = cv2.getTrackbarPos('max_H', 'Track Bars')
            max_s = cv2.getTrackbarPos('max_S', 'Track Bars')
            max_v = cv2.getTrackbarPos('max_V', 'Track Bars')

            mask = cv2.inRange(img_hsv, (min_h, min_s, min_v), (max_h, max_s, max_v))

            cv2.imshow('Mask Image', mask)

            key = cv2.waitKey(25)
            if key == ord('q'):
                break

            print('min_h', 'min_s', 'min_v')
            print(min_h, min_s, min_v)
            print('max_h', 'max_s', 'max_v')
            print(max_h, max_s, max_v)

        self.min_h = min_h
        self.min_s = min_s
        self.min_v = min_v

        self.max_h = max_h
        self.max_s = max_s
        self.max_v = max_v

    def get_mask(self, img):
        """
        Get a mask based on the HSV limits. Perform image processing to clean up mask
        :param img: raw image
        :return:
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.inRange(img, (self.min_h, self.min_s, self.min_v), (self.max_h, self.max_s, self.max_v))

        return self.reduce_noise(mask=img)

    # @staticmethod
    # def erode(img):
    #     return cv2.erode(img, kernel=(3, 3), iterations=10)
    #
    # @staticmethod
    # def dilate(img):
    #     return cv2.dilate(img, kernel=(5, 5), iterations=10)

    @staticmethod
    def reduce_noise(mask):
        """
        Clean up mask. Remove noise
        :param mask:
        :return:
        """
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)

        return mask

    @staticmethod
    def find_contours(mask, img):
        """

        :param mask: masked image used to find contours
        :param img: raw image used to plot contours
        :return:
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        return contours, img

    def get_point_cloud(self, images, masks, show=True):
        point_cloud_data = []
        layer_count = 0

        layers = zip(images, masks, self.img_paths)

        for layer in layers:
            layer_count += 1

            contours, img_annotated = self.find_contours(mask=layer[1], img=layer[0])

            # todo make layer selection more robust
            layer = int(layer[2].stem[-2:])
            for cnt in contours:
                for index, point in enumerate(cnt):
                    point_3d = np.append(point[0], layer_count)
                    point_cloud_data.append(point_3d)

        point_cloud = pv.PolyData(point_cloud_data)

        if show:
            point_cloud.plot(eye_dome_lighting=True)

        return point_cloud

    # @staticmethod
    # def convert_point_cloud_to_mesh(point_cloud, show=True):
    #     mesh = point_cloud.reconstruct_surface(nbr_sz=10)
    #     mesh.save('mesh_1.stl')
    #
    #     if show:
    #         mesh.plot(color='orange')
    #
    #     return mesh

    def calculate_layer_volume(self, mask):
        """
        Calculates the volume of a single layer based on image mask
        area_px_to_area_um converts px area to um area
        :param mask: mask of the OPC stack layer
        :return: volume of layer in um^3
        """
        area_px = np.count_nonzero(mask)        # px^2
        area_px_to_area_um = self.x_um_per_px * self.y_um_per_px    # um^2 / px^2
        area_um = area_px * area_px_to_area_um  # um^2

        volume = area_um * self.z_um_per_layer  # um^3

        return volume


    # def convert_data_to_bool_array(self, img, layer_count: int, data):
    #     len_x = img.shape[0]
    #     len_y = img.shape[1]
    #     len_z = layer_count + 1     # layer count starts at 1
    #
    #     bool_array = np.zeros((len_x, len_y, len_z), dtype=bool)
    #
    #     for point in data:
    #         x, y, z = point
    #         bool_array[int(x), int(y), int(z)] = True
    #
    #     return bool_array

    def show_image(self, img, delay=0, name=None):
        name = name or 'image'
        cv2.imshow(name, img)
        cv2.waitKey(delay)
        cv2.destroyAllWindows()

    def select_roi(self, img):
        self.show_image(img=img, name='select ROI')
        x, y, w, h = cv2.selectROI(windowName='select ROI', img=img, showCrosshair=False, fromCenter=False)

        roi_mask = np.zeros(shape=img.shape, dtype='uint8')
        roi_mask[y:y+h, x:x+w] = 255

        self.roi = roi_mask

    def get_volume_from_stack(self, stack_name: str):
        directory = Path.cwd()
        self.img_paths = list(Path(directory, 'OPC_stacks', stack_name).glob('*.tif'))

        if len(self.img_paths) == 0:
            return 0

        # point_cloud_data = []
        # layer_count = 0

        mask_of_all_layers = None
        images = []
        masks = []
        total_volume = 0

        for path in self.img_paths:
            # layer_count += 1

            img = cv2.imread(str(path))

            mask = self.get_mask(img)

            masks.append(mask)
            images.append(img)

            if mask_of_all_layers is None:
                mask_of_all_layers = mask
            else:
                mask_of_all_layers |= mask

            # contours, img_annotated = self.find_contours(mask=mask, img=img)
            #
            # # todo make layer selection more robust
            # layer = int(path.stem[-2:])
            # for cnt in contours:
            #     for index, point in enumerate(cnt):
            #         point_3d = np.append(point[0], layer)
            #         point_cloud_data.append(point_3d)

        point_cloud = self.get_point_cloud(images=images, masks=masks, show=True)

        self.select_roi(mask_of_all_layers)
        cv2.destroyAllWindows()

        mask_of_all_layers &= self.roi
        self.show_image(mask_of_all_layers)

        for index, mask in enumerate(masks):
            mask &= self.roi
            masks[index] = mask
            total_volume += self.calculate_layer_volume(mask=mask)

        point_cloud = self.get_point_cloud(images=images, masks=masks, show=True)

        print(f'Cell volume = {total_volume} um^3')

        print('done')
