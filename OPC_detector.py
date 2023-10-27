import cv2
import numpy as np
import pyvista as pv
from pathlib import Path
# import PVGeo
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

# x=0.07um/pixel, y=0.07um/pixel, z=1.0um/layer


class OPCDetector:
    def __init__(self):
        self.img_paths = ...
        self.original_img_shape = ...

        self.min_h = 0       # 0
        self.min_s = 1       # 15
        self.min_v = 1       # 12

        self.max_h = 18       # 12, 25
        self.max_s = 255       # 255
        self.max_v = 149       # 255

        self.x_um_per_px = 0.07
        self.y_um_per_px = 0.07
        self.z_um_per_layer = 0.1

        self.shrink_iterations = 6
        self.shrink_include_threshold = 0.2

        self.roi = ...

    @staticmethod
    def do_nothing(*args):
        pass

    def find_hsv_values(self, img):
        cv2.namedWindow('Track Bars', cv2.WINDOW_NORMAL)

        cv2.createTrackbar('min_H', 'Track Bars', 0, 40, self.do_nothing)
        cv2.createTrackbar('min_S', 'Track Bars', 1, 255, self.do_nothing)
        cv2.createTrackbar('min_V', 'Track Bars', 1, 255, self.do_nothing)

        cv2.createTrackbar('max_H', 'Track Bars', 0, 40, self.do_nothing)   # 180 is max
        cv2.createTrackbar('max_S', 'Track Bars', 1, 255, self.do_nothing)
        cv2.createTrackbar('max_V', 'Track Bars', 1, 255, self.do_nothing)

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

    def get_mask(self, img, show:bool = False):
        """
        Get a mask based on the HSV limits. Perform image processing to clean up mask
        :param img: raw image
        :param show: show image or not
        :return:
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = cv2.inRange(img, (self.min_h, self.min_s, self.min_v), (self.max_h, self.max_s, self.max_v))

        img = self.reduce_noise(mask=img)

        if show:
            self.show_image(img=img)

        return img

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

        # convert z scale to match x,y scale
        z_scale_factor = int(round(self.z_um_per_layer / self.x_um_per_px, 0))

        for layer in layers:
            contours, img_annotated = self.find_contours(mask=layer[1], img=layer[0])

            # self.show_image(img_annotated)

            # for interpolation in range(z_scale_factor):
            for interpolation in range(1):
                for cnt in contours:
                    for index, point in enumerate(cnt):
                        point_3d = np.append(point[0], layer_count + interpolation)
                        point_cloud_data.append(point_3d)

            layer_count += z_scale_factor

        point_cloud = pv.PolyData(point_cloud_data)

        if show:
            point_cloud.plot(eye_dome_lighting=True)

        return point_cloud

    @staticmethod
    def convert_point_cloud_to_mesh(point_cloud, show=True):
        mesh = point_cloud.reconstruct_surface(nbr_sz=10)
        mesh.save('mesh_1.stl')

        if show:
            mesh.plot(color='orange')

        return mesh

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
        img = img * 255
        self.show_image(img=img, name='select ROI')
        x, y, w, h = cv2.selectROI(windowName='select ROI', img=img, showCrosshair=False, fromCenter=False)

        roi_mask = np.zeros(shape=img.shape, dtype='uint8')
        roi_mask[y:y+h, x:x+w] = 255

        self.roi = roi_mask

    # https: // stackoverflow.com / questions / 10685654 / reduce - resolution - of - array - through - summation
    def shrink(self, data, iterations: int):
        """
        Finds the multiples of the data size.
        Creates a smaller array by averaging the original data array.
        :param data:
        :param iterations: How many times to shrink the data. Occurs in multiples of data size
        :return:
        """
        multiples = [i for i in range(1, data.shape[0] + 1) if data.shape[0] % i == 0]
        max_iterations = len(multiples) - 1
        if iterations > max_iterations:
            iterations = max_iterations
        rows = multiples[-(iterations + 1)]    # +1 because last element in array is original size
        cols = rows
        return data.reshape(rows, int(data.shape[0] / rows), cols, int(data.shape[1] / cols)).mean(axis=3).mean(axis=1).astype(np.uint8)

    # https: // stackoverflow.com / questions / 42611342 / representing - voxels - with-matplotlib
    def plot_voxels(self, masks):
        x_and_y_scale_factor = int(self.original_img_shape[0] / masks[0].shape[0])
        # prepare some coordinates
        x_dim = masks[0].shape[0]
        y_dim = x_dim
        z_dim = len(masks)
        x, y, z = np.indices((z_dim+1, x_dim+1, y_dim+1))

        voxelarray = np.zeros((z_dim, x_dim, y_dim))

        for layer, mask in enumerate(masks):
            voxelarray[layer] = mask

        # data[0][10:50][10:50] = 1

        # draw cuboids in the top left and bottom right corners, and a link between
        # them
        # cube1 = (x < 3) & (y < 3) & (z < 3)
        # cube2 = (x >= (dims-3)) & (y >= (dims-3)) & (z >= (dims-3))
        # link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

        # combine the objects into a single boolean array
        # voxelarray = cube1 | cube2 | link

        #
        # # set the colors of each object
        colors = np.empty(voxelarray.shape, dtype=object)
        # colors[link] = 'red'
        # colors[cube1] = 'blue'
        # colors[cube2] = 'green'
        colors[voxelarray > 0] = 'red'

        # and plot everything
        ax = plt.figure().add_subplot(projection='3d')
        # ax.voxels(voxelarray, facecolors=colors, edgecolor='k')
        # ax.voxels(voxelarray, facecolors=colors)
        # ax.voxels(z*self.z_um_per_layer, x, y, voxelarray)
        ax.voxels(x*self.z_um_per_layer, y*x_and_y_scale_factor*self.x_um_per_px, z*x_and_y_scale_factor*self.y_um_per_px, voxelarray)
        ax.set_xlabel('z')     # '0 - Dim'
        ax.set_ylabel('x')     # '1 - Dim'
        ax.set_zlabel('y')     # '2 - Dim'

        # ax.plot_surface(x, y, z)

        ax.set_aspect('equal')

        plt.show()

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
            img = cv2.imread(str(path))

            mask = self.get_mask(img)

            self.original_img_shape = mask.shape

            mask = self.shrink(mask, iterations=self.shrink_iterations)
            mask = mask / 255
            mask = (mask > self.shrink_include_threshold).astype(np.uint8)

            masks.append(mask)
            images.append(img)

            if mask_of_all_layers is None:
                mask_of_all_layers = mask
            else:
                mask_of_all_layers |= mask

        self.plot_voxels(masks)

        # point_cloud = self.get_point_cloud(images=images, masks=masks, show=True)

        for mask in masks:
            self.show_image(mask*255)

        self.select_roi(mask_of_all_layers)
        cv2.destroyAllWindows()

        mask_of_all_layers &= self.roi

        self.show_image(mask_of_all_layers)

        for index, mask in enumerate(masks):
            mask &= self.roi
            masks[index] = mask
            # self.show_image(img=mask)
            total_volume += self.calculate_layer_volume(mask=mask)

        point_cloud = self.get_point_cloud(images=images, masks=masks, show=True)

        print(f'Cell volume = {int(total_volume)} um^3')

        print('done')
