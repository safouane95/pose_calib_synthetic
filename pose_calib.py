import numpy as np

from functions_ import *
from ximea import xiapi


def main():
    parser = argparse.ArgumentParser(description="Interactive camera calibration using efficient pose selection")
    parser.add_argument("-c", "--config", help="path to calibration configuration (e.g. data/calib_config.yml)")
    parser.add_argument("-o", "--outfile", help="path to calibration output (defaults to calib_<cameraId>.yml)")
    parser.add_argument("-m", "--mirror", action="store_true", help="horizontally flip the camera image for display")
    args = parser.parse_args()

    if args.config is None:
        print("falling back to " + sys.path[0] + "/data/calib_config.yml")
        args.config = sys.path[0] + "/data/calib_config.yml"

    cfg = cv2.FileStorage(args.config, cv2.FILE_STORAGE_READ)
    assert cfg.isOpened()

    calib_name = "default"
    if not cfg.getNode("v4l_id").empty():
        calib_name = cfg.getNode("v4l_id").string()

    # Video I/O
    live = cfg.getNode("images").empty()
    if live:
        cap = UVCVideoCapture(cfg)
        #add_camera_controls("PoseCalib", cap)
        wait = 1
    else:
        cv2.namedWindow("PoseCalib")
        cap = cv2.VideoCapture(cfg.getNode("images").string() + "frame%0d.png", cv2.CAP_IMAGES)
        wait = 0
        assert cap.isOpened()

    tracker = ChArucoDetector(cfg)

    # user guidance
    ugui = UserGuidance(tracker, cfg.getNode("terminate_var").real())

    # runtime variables
    mirror = False
    save = False

    while True:
        force = not live  # force add frame to calibration

        status, _img = cap.read()
        ############################# TODO: REMPLACE _IMG PAR IMAGE VIRTUELLE
        f_u = 1600  # mm to pixel !
        f_v = 1600  # mm to pixel !
        c_u = 640  # pixels
        c_v = 512  # pixels
        square_len = 6  # mm
        nk = 1  # number of keyframes
        #K_ideal = np.array([f_u, 0., c_u, 0., f_v, c_v, 0., 0., 1.])
        K_ideal = np.column_stack((1600, 0, 640, 0, 1600, 512, 0, 0, 1)).reshape(-1,3,3)
        distor = np.array([-0.05, 0.05, 0, 0, 0])
        '''#tgt_param = index_of_dispersion.argmax()  # parameter to optimise
        board_units = np.array([tracker.board_sz[0], tracker.board_sz[1], tracker.board_sz[0]]) * square_len
        
        tgt_r, tgt_t = PoseGeneratorDist.get_pose(board_units,
                                                  nk,
                                                  1,
                                                  K_ideal,
                                                  distor)
        _img = BoardPreview.create_maps(K_ideal, distor, tracker.img_size)
        _img = _img.project(tgt_r, tgt_t)'''

        board_units = np.array([tracker.board_sz[0], tracker.board_sz[1], tracker.board_sz[0]]) * 6
        r, t = oribital_pose(board_units, 0, np.pi / 4, 1.6, np.pi / 8)
        # print(r, t)
        board = BoardPreview(tracker.board.draw(tuple(tracker.board_sz * 12)))
        img_size = (int(cfg.getNode("image_width").real()), int(cfg.getNode("image_height").real()))
        board.create_maps(K_ideal, distor, img_size)
        board_warped = board.project(r, t)
        img = board_warped
        cv2.imshow('img', img)
        '''cdist = distor
        if cdist is None:
            cdist = np.array([0., 0., 0., 0.])

        scale = np.diag((_img.shape[0] / _img.shape[0], _img.shape[1] / _img.shape[1], 1))

        #Knew = cv2.getOptimalNewCameraMatrix(K_ideal, cdist, _img.shape, 1)[0]
        #maps = make_distort_map(K_ideal, _img.shape, cdist, Knew)
        inter = cv2.INTER_NEAREST
        img = _img
        shadow = np.ones(img.shape[:2], dtype=np.uint8)
        img = project_img(shadow if shadow else img, _img.shape, K_ideal, r, t)
        img = cv2.remap(img, 0.05, -0.05, inter)
        img = cv2.resize(img, _img.shape, interpolation=inter)

        testing = BoardPreview.create_maps(K_ideal, distor, img.shape)
        print(testing)

        board = BoardPreview(tracker.board.draw(tuple(tracker.board_sz * 12)))

        img1 = board.create_maps(K_ideal, distor, tracker.img_size)
        cv2.imshow('image1', img1)'''

        ######################################################## TODO: END

        if status:
            img = _img
        else:
            force = False

        tracker.detect(img)

        if save:
            save = False
            force = True

        out = img.copy()

        ugui.draw(out, mirror)

        ugui.update(force)

        if ugui.converged:
            if args.outfile is None:
                outfile = "calib_{}.yml".format(calib_name)
            else:
                outfile = args.outfile
            ugui.write(outfile)

        if ugui.user_info_text:
            cv2.displayOverlay("PoseCalib", ugui.user_info_text, 1000 // 30)

        cv2.imshow("PoseCalib", out)
        k = cv2.waitKey(wait)

        if k == 27:
            break
        elif k == ord('m'):
            mirror = not mirror
        elif k == ord('c'):
            save = True


if __name__ == "__main__":
    main()
