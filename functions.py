import argparse
import sys
import cv2
import datetime
import numpy as np
import time

import cv2
from cv2.aruco import Dictionary_get, CharucoBoard_create, drawAxis, interpolateCornersCharuco, detectMarkers, estimatePoseCharucoBoard

import numpy.linalg as la
from ximea import xiapi


def gen_bin(s, e):
    """
    generate values in range by binary subdivision
    """
    t = (s + e) / 2
    lst = [(s, t), (t, e)]

    while lst:
        s, e = lst.pop(0)
        t = (s + e) / 2
        lst.append((s, t))
        lst.append((t, e))
        yield t


def unproject(p, K, cdist, Z):
    """
    project pixel back to a 3D coordinate at depth Z
    """
    p = cv2.undistortPoints(p.reshape(-1, 1, 2), K, cdist).ravel()
    return np.array([p[0], p[1], 1]) * Z


def oribital_pose(bbox, rx, ry, Z, rz=0):
    """
    @param bbox: object bounding box. note: assumes planar object with virtual Z dimension.
    @param rx: rotation around x axis in rad
    @param ry: rotation around y axis in rad
    @param Z: distance to camera in board lengths
    @return: rvec, tvec
    """
    Rz = cv2.Rodrigues(np.array([0., 0., rz]))[0]
    Rx = cv2.Rodrigues(np.array([np.pi + rx, 0., 0.]))[0]  # flip by 180° so Z is up
    Ry = cv2.Rodrigues(np.array([0., ry, 0.]))[0]

    R = np.eye(4)
    R[:3, :3] = (Ry).dot(Rx).dot(Rz)

    # translate board to its center
    Tc = np.eye(4)
    Tc[3, :3] = R[:3, :3].dot(bbox * [-0.5, -0.5, 0])

    # translate board to center of image
    T = np.eye(4)
    T[3, :3] = bbox * [-0.5, -0.5, Z]

    # rotate center of board
    Rf = la.inv(Tc).dot(R).dot(Tc).dot(T)

    return cv2.Rodrigues(Rf[:3, :3])[0].ravel(), Rf[3, :3]

def pose_planar_fullscreen(K, cdist, img_size, bbox):
    KB = K.dot([bbox[0], bbox[1], 0])  # ignore principal point
    Z = (KB[0:2] / img_size).min()
    pB = KB / Z

    r = np.array([np.pi, 0, 0])  # flip image
    # move board to center, org = bl
    p = np.array([img_size[0] / 2 - pB[0] / 2, img_size[1] / 2 + pB[1] / 2])
    t = unproject(p, K, cdist, Z)
    return r, t

def pose_from_bounds(src_ext, tgt_rect, K, cdist, img_sz):
    rot90 = tgt_rect[3] > tgt_rect[2]

    MIN_WIDTH = img_sz[0] // 3.333

    if rot90:
        src_ext = src_ext.copy()
        src_ext[0], src_ext[1] = src_ext[1], src_ext[0]

        if tgt_rect[3] < MIN_WIDTH:
            scale = MIN_WIDTH / tgt_rect[2]
            tgt_rect[3] = MIN_WIDTH
            tgt_rect[2] *= scale
    else:
        if tgt_rect[2] < MIN_WIDTH:
            scale = MIN_WIDTH / tgt_rect[2]
            tgt_rect[2] = MIN_WIDTH
            tgt_rect[3] *= scale

    aspect = src_ext[0] / src_ext[1]

    # match aspect ratio of tgt to src, but keep tl
    if not rot90:
        # adapt height
        tgt_rect[3] = tgt_rect[2] / aspect
    else:
        # adapt width
        tgt_rect[2] = tgt_rect[3] * aspect

    r = np.array([np.pi, 0, 0])

    # org is bl
    if rot90:
        R = cv2.Rodrigues(r)[0]
        Rz = cv2.Rodrigues(np.array([0., 0., -np.pi / 2]))[0]
        R = R.dot(Rz)
        r = cv2.Rodrigues(R)[0].ravel()
        # org is tl

    Z = (K[0, 0] * src_ext[0]) / tgt_rect[2]

    # clip to image region
    max_off = img_sz - tgt_rect[2:4]
    tgt_rect[0:2] = tgt_rect[0:2].clip([0, 0], max_off)

    if not rot90:
        tgt_rect[1] += tgt_rect[3]

    t = unproject(np.array([tgt_rect[0], tgt_rect[1]], dtype=np.float32), K, cdist, Z)

    if not rot90:
        tgt_rect[1] -= tgt_rect[3]

    return r, t, tgt_rect

class PoseGeneratorDist:
    """
    generate poses based on min/ max distortion
    """
    SUBSAMPLE = 20

    def __init__(self, img_size):
        self.img_size = img_size

        self.stats = [1, 1]  # number of (intrinsic, distortion) poses

        self.orbitalZ = 1.6
        rz = np.pi / 8

        # valid poses:
        # r_x, r_y -> -70° .. 70°
        self.orbital = (
            gen_bin(np.array([-(70 / 180) * np.pi, 0, self.orbitalZ, rz]),
                    np.array([(70 / 180) * np.pi, 0, self.orbitalZ, rz])),
            gen_bin(np.array([0, -(70 / 180) * np.pi, self.orbitalZ, rz]),
                    np.array([0, (70 / 180) * np.pi, self.orbitalZ, rz]))
        )

        self.mask = np.zeros(np.array(img_size) // self.SUBSAMPLE, dtype=np.uint8).T
        self.sgn = 1

    def compute_distortion(self, K, cdist, subsample=1):
        return sparse_undistort_map(K, self.img_size, cdist, K, subsample)

    def get_pose(self, bbox, nk, tgt_param, K, cdist):
        """
        @param bbox: bounding box of the calibration pfrom ximea import xiapi
attern
        @param nk: number of keyframes captured so far
        @param tgt_param: parameter that should be optimized by the pose
        @param K, cdist: current calibration estimate
        """
        if nk == 0:
            # init sequence: first keyframe 45° tilted to camera
            return oribital_pose(bbox, 0, np.pi / 4, self.orbitalZ, np.pi / 8)
        if nk == 1:
            # init sequence: second keyframe
            return pose_planar_fullscreen(K, cdist, self.img_size, bbox)
        if tgt_param < 4:
            # orbital pose is used for focal length
            axis = (tgt_param + 1) % 2  # f_y -> r_x

            self.stats[0] += 1
            r, t = oribital_pose(bbox, *next(self.orbital[axis]))

            if tgt_param > 1:
                off = K[:2, 2].copy()
                off[tgt_param - 2] += self.img_size[tgt_param - 2] * 0.05 * self.sgn
                off3d = unproject(off, K, cdist, t[2])
                off3d[2] = 0
                t += off3d
                self.sgn *= -1

            return r, t

        dpts, pts = self.compute_distortion(K, cdist, self.SUBSAMPLE)

        bounds = loc_from_dist(pts, dpts, mask=self.mask)[0]

        if bounds is None:
            # FIXME: anything else?
            print("loc_from_dist failed. return orbital pose instead of crashing")
            return self.get_pose(bbox, nk, 3, axis, K, cdist)

        self.stats[1] += 1
        r, t, nbounds = pose_from_bounds(bbox, bounds * self.SUBSAMPLE, K, cdist, self.img_size)
        x, y, w, h = np.ceil(np.array(nbounds) / self.SUBSAMPLE).astype(int)
        self.mask[y:y + h, x:x + w] = 1

        return r, t

def get_bounds(thresh, mask):
    MAX_OVERLAP = 0.9
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    # look for the largest object that is not masked
    while contours:
        mx = np.argmax([len(c) for c in contours])
        contour = contours[mx]
        aabb = cv2.boundingRect(contour)

        x, y, w, h = aabb
        if mask is not None and (cv2.countNonZero(mask[y:y + h, x:x + w]) / (w * h) > MAX_OVERLAP):
            del contours[mx]  # remove from candidates
            continue

        return (aabb, contour)

    return None

def make_distort_map(K, sz, dist, Knew):
    """
    creates a map for distorting an image as a opposed to the default
    behaviour of undistorting
    @param sz: width, height
    """
    pts = np.array(np.meshgrid(range(sz[0]), range(sz[1]))).T.reshape(-1, 1, 2)
    dpts = cv2.undistortPoints(pts.astype(np.float32), K, dist, P=Knew)

    return dpts.reshape(sz[0], sz[1], 2).T

def sparse_undistort_map(K, sz, dist, Knew, step=1):
    """
    same output as initUndistortRectifyMap, but sparse
    @param sz: width, height
    @return: distorted points, original points
    """
    zero = np.zeros(3)
    pts = np.array(np.meshgrid(range(0, sz[0], step), range(0, sz[1], step))).T.reshape(-1, 1, 2)

    if step == 1:
        dpts = cv2.initUndistortRectifyMap(K, dist, None, Knew, sz, cv2.CV_32FC2)[0].transpose(1, 0, 2)
    else:
        pts3d = cv2.undistortPoints(pts.astype(np.float32), Knew, None)
        pts3d = cv2.convertPointsToHomogeneous(pts3d).reshape(-1, 3)
        dpts = cv2.projectPoints(pts3d, zero, zero, K, dist)[0]

    shape = (sz[0] // step, sz[1] // step, 2)

    return dpts.reshape(-1, 2).reshape(shape), pts.reshape(shape)

def get_diff_heatmap(img1, img2, colormap=True):
    """
    creates a heatmap from two point images
    """
    sz = img1.shape[:2]
    l2diff = la.norm((img1 - img2).reshape(-1, 2), axis=1).reshape(sz).T

    if colormap:
        l2diff = cv2.normalize(l2diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        l2diff = cv2.applyColorMap(l2diff, cv2.COLORMAP_JET)

    return l2diff, l2diff.max()

def loc_from_dist(pts, dpts, mask=None, lower=False, thres=1.0):
    """
    compute location based on distortion strength
    @param pts: sampling locations
    @param dpts: distorted points
    @param mask: mask for ignoring locations
    @param lower: find location with minimal distortion instead
    @param thres: distortion strength to use as threshold [%]
    """
    diff = la.norm((pts - dpts).reshape(-1, 2), axis=1)
    diff = diff.reshape(pts.shape[0:2]).T
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    bounds = None

    while not bounds and thres >= 0 and thres <= 1:
        if lower:
            thres += 0.05
            thres_img = cv2.threshold(diff, thres * 255, 255, cv2.THRESH_BINARY_INV)[1]
        else:
            thres -= 0.05
            thres_img = cv2.threshold(diff, thres * 255, 255, cv2.THRESH_BINARY)[1]

        bounds = get_bounds(thres_img, mask)

        if bounds is None:
            continue

        # ensure area is not 0
        if bounds[0][2] * bounds[0][3] == 0:
            bounds = None

    if bounds is None:
        return None, None

    return np.array(bounds[0]), thres_img

def project_img(img, sz, K, rvec, t, flags=cv2.INTER_LINEAR):
    """
    projects a 2D object (image) according to parameters
    @param img: image to project
    @param sz: size of the final image
    """
    # construct homography
    R = cv2.Rodrigues(rvec)[0]
    H = K.dot(np.array([R[:, 0], R[:, 1], t]).T)
    H /= H[2, 2]

    return cv2.warpPerspective(img, H, sz, flags=flags)

class BoardPreview:
    SIZE = (640, 480)

    def __init__(self, img):
        # generate styled board image
        self.img = img
        self.img = cv2.flip(self.img, 0)  # flipped when printing
        self.img[self.img == 0] = 64  # set black to gray
        self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        self.img[:, :, 0::2] = 0  # set red and blue to zero

        self.shadow = np.ones(self.img.shape[:2], dtype=np.uint8)  # used for overlap score

    def create_maps(self, K, cdist, sz):
        if cdist is None:
            cdist = np.array([0., 0., 0., 0.])

        self.sz = sz
        scale = np.diag((self.SIZE[0] / sz[0], self.SIZE[1] / sz[1], 1))
        K = scale.dot(K)

        sz = self.SIZE
        self.Knew = cv2.getOptimalNewCameraMatrix(K, cdist, sz, 1)[0]
        self.maps = make_distort_map(K, sz, cdist, self.Knew)

    def project(self, r, t, shadow=False, inter=cv2.INTER_NEAREST):
        img = project_img(self.shadow if shadow else self.img, self.SIZE, self.Knew, r, t)
        img = cv2.remap(img, self.maps[0], self.maps[1], inter)
        img = cv2.resize(img, self.sz, interpolation=inter)
        return img

class ChArucoDetector:
    def __init__(self, cfg):
        # configuration
        self.board_sz = np.array([int(cfg.getNode("board_x").real()), int(cfg.getNode("board_y").real())])
        self.square_len = cfg.getNode("square_len").real()
        self.ardict = Dictionary_get(int(cfg.getNode("dictionary").real()))

        marker_len = cfg.getNode("marker_len").real()
        self.board = CharucoBoard_create(self.board_sz[0], self.board_sz[1], self.square_len, marker_len, self.ardict)
        self.img_size = (int(cfg.getNode("image_width").real()), int(cfg.getNode("image_height").real()))

        # per frame data
        self.N_pts = 0
        self.pose_valid = False
        self.raw_img = None
        self.pt_min_markers = int(cfg.getNode("pt_min_markers").real())

        self.intrinsic_valid = False

        # optical flow calculation
        self.last_ccorners = None
        self.last_cids = None
        # mean flow if same corners are detected in consecutive frames
        self.mean_flow = None

    def set_intrinsics(self, calib):
        self.intrinsic_valid = True
        self.K = calib.K
        self.cdist = calib.cdist

    def draw_axis(self, img):
        drawAxis(img, self.K, self.cdist, self.rvec, self.tvec, self.square_len)

    def detect_pts(self, img):
        self.corners, ids, self.rejected = detectMarkers(img, self.ardict)

        self.N_pts = 0
        self.mean_flow = None

        if ids is None or ids.size == 0:
            self.last_ccorners = None
            self.last_cids = None
            return

        res = interpolateCornersCharuco(self.corners, ids, img, self.board, minMarkers=self.pt_min_markers)
        self.N_pts, self.ccorners, self.cids = res

        if self.N_pts == 0:
            return

        if not np.array_equal(self.last_cids, self.cids):
            self.last_ccorners = self.ccorners.reshape(-1, 2)
            self.last_cids = self.cids
            return

        diff = self.last_ccorners - self.ccorners.reshape(-1, 2)
        self.mean_flow = np.mean(la.norm(diff, axis=1))
        self.last_ccorners = self.ccorners.reshape(-1, 2)
        self.last_cids = self.cids

    def detect(self, img):
        self.raw_img = img.copy()
        self.detect_pts(img)

        if self.intrinsic_valid:
            self.update_pose()

    def get_pts3d(self):
        return self.board.chessboardCorners[self.cids].reshape(-1, 3)

    def get_calib_pts(self):
        return (self.ccorners.copy(), self.get_pts3d())

    def update_pose(self):
        if self.N_pts < 4:
            self.pose_valid = False
            return

        ret = estimatePoseCharucoBoard(self.ccorners, self.cids, self.board, self.K, self.cdist)
        self.pose_valid, rvec, tvec = ret

        if not self.pose_valid:
            return

        self.rvec = rvec.ravel()
        self.tvec = tvec.ravel()

        # print(cv2.RQDecomp3x3(cv2.Rodrigues(self.rvec)[0])[0])
        # print(self.tvec)

class Calibrator:
    def __init__(self, img_size):
        self.img_size = img_size
        self.nintr = 9
        self.unknowns = None  # number of unknowns in our equation system

        # initial K matrix
        # with aspect ratio of 1 and pp at center. Focal length is empirical.
        self.Kin = cv2.getDefaultNewCameraMatrix(np.diag([1000, 1000, 1]), img_size, True)
        self.K = self.Kin.copy()
        self.cdist = None
        self.flags = cv2.CALIB_USE_LU

        # calibration data
        self.keyframes = []
        self.reperr = float("NaN")
        self.PCov = np.zeros((self.nintr, self.nintr))  # parameter covariance
        self.pose_var = np.zeros(6)
        self.disp_idx = None  # index of dispersion

    def get_intrinsics(self):
        K = self.K
        return [K[0, 0], K[1, 1], K[0, 2], K[1, 2]] + list(self.cdist.ravel())

    def calibrate(self, keyframes=None):
        flags = self.flags

        if not keyframes:
            keyframes = self.keyframes

        assert (keyframes)

        nkeyframes = len(keyframes)

        if nkeyframes <= 1:
            # restrict first calibration to K matrix parameters
            flags |= cv2.CALIB_FIX_ASPECT_RATIO

        if nkeyframes <= 1:
            # with only one frame we just estimate the focal length
            flags |= cv2.CALIB_FIX_PRINCIPAL_POINT

            flags |= cv2.CALIB_ZERO_TANGENT_DIST
            flags |= cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3

        t = time.process_time()

        res = calibrateCamera(keyframes, self.img_size, flags, self.Kin)

        self.reperr, self.K, self.cdist, rvecs, tvecs, self.PCov, self.N_pts = res

        self.calib_t = time.process_time() - t

        self.pose_var = compute_pose_var(rvecs, tvecs)
        self.unknowns = self.nintr + 6 * nkeyframes

        pvar = np.diag(self.PCov)
        self.mean_extr_var = mean_extr_var(pvar[self.nintr:])

        self.disp_idx = index_of_dispersion(self.get_intrinsics(), np.diag(self.PCov)[:self.nintr])
        return self.disp_idx

def index_of_dispersion(mean, var):
    """
    computes index of dispersion:
    https://en.wikipedia.org/wiki/Index_of_dispersion
    """
    return var / [abs(v) if abs(v) > 0 else 1. for v in mean]

def mean_extr_var(var):
    """
    computes the mean of the extrinsic variances
    @param var: variance vector excluding the intrinsic parameters
    """
    assert (len(var) % 6 == 0)
    nframes = len(var) // 6
    my_var = var[:6].copy()
    for i in range(1, nframes - 1):
        my_var += var[6 * i:6 * (i + 1)]
    return my_var / nframes

def estimate_pt_std(res, d, n):
    """
    estimate the accuracy of point measurements given the reprojection error
    @param res: the reprojection error
    """
    return res / np.sqrt(1 - d / (2 * n))

def Jc2J(Jc, N_pts, nintr=9):
    """
    decompose a compact 'single view' jacobian into a sparse 'multi view' jacobian
    @param Jc: compact single view jacobian
    @param N_pts: number of points per view
    @param nintr: number of intrinsic parameters
    """
    total = np.sum(N_pts)

    J = np.zeros((total * 2, nintr + 6 * len(N_pts)))
    J[:, :nintr] = Jc[:, 6:]

    i = 0

    for j, n in enumerate(N_pts):
        J[2 * i:2 * i + 2 * n, nintr + 6 * j:nintr + 6 * (j + 1)] = Jc[2 * i:2 * i + 2 * n, :6]
        i += n
    return J

def compute_pose_var(rvecs, tvecs):
    ret = np.empty(6)
    reuler = np.array([cv2.RQDecomp3x3(cv2.Rodrigues(r)[0])[0] for r in rvecs])

    # workaround for the given board so r_x does not oscilate between +-180°
    reuler[:, 0] = reuler[:, 0] % 360

    ret[0:3] = np.var(reuler, axis=0)
    ret[3:6] = np.var(np.array(tvecs) / 10, axis=0).ravel()  # [mm]
    return ret

def compute_state_cov(pts3d, rvecs, tvecs, K, cdist, flags):
    """
    state covariance from current intrinsic and extrinsic estimate
    """
    P_cam = []
    N_pts = [len(pts) for pts in pts3d]

    # convert to camera coordinate system
    for i in range(len(pts3d)):
        R = cv2.Rodrigues(rvecs[i])[0]
        P_cam.extend([R.dot(P) + tvecs[i].ravel() for P in pts3d[i]])

    zero = np.array([0, 0, 0.], dtype=np.float32)

    # get jacobian
    Jc = cv2.projectPoints(np.array(P_cam), zero, zero, K, cdist)[1]
    J = Jc2J(Jc, N_pts)
    JtJ = J.T.dot(J)

    if flags & (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3):
        # TODO: remove the according fixed rows so we can invert this
        return np.zeros_like(JtJ)
    return la.inv(JtJ)

def calibrateCamera(keyframes, img_size, flags, K):
    pts2d = []
    pts3d = []
    N = 0

    for p2d, p3d in keyframes:
        pts2d.append(p2d)
        pts3d.append(p3d)
        N += len(p2d)

    res = cv2.calibrateCamera(np.array(pts3d), np.array(pts2d), img_size, K, None, flags=flags)

    reperr, K, cdist, rvecs, tvecs = res
    cov = compute_state_cov(pts3d, rvecs, tvecs, K, cdist, flags)
    return reperr, K, cdist, rvecs, tvecs, cov, N

def debug_jaccard(img, tmp):
    dbg = img.copy() + tmp * 2
    cv2.imshow("jaccard", dbg * 127)

class UserGuidance:
    AX_NAMES = ("red", "green", "blue")
    INTRINSICS = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")
    POSE = ("rx", "ry", "rz", "tx", "ty", "tz")

    SQUARE_LEN_PIX = 12
    # parameters that are optimized by the same board poses
    PARAM_GROUPS = [(0, 1, 2, 3), (4, 5, 6, 7, 8)]

    def __init__(self, tracker, var_terminate=0.1):
        # get geometry from tracker
        self.tracker = tracker
        self.allpts = np.prod(tracker.board_sz - 1)
        self.square_len = tracker.board.getSquareLength()
        self.SQUARE_LEN_PIX = int(self.square_len)

        self.img_size = tracker.img_size

        self.overlap = np.zeros((self.img_size[1], self.img_size[0]), dtype=np.uint8)

        # preview image
        self.board = BoardPreview(self.tracker.board.draw(tuple(tracker.board_sz * self.SQUARE_LEN_PIX)))

        self.calib = Calibrator(tracker.img_size)
        self.min_reperr_init = float("inf")

        # desired pose of board for first frame
        # translation defined in terms of board dimensions
        self.board_units = np.array([tracker.board_sz[0], tracker.board_sz[1], tracker.board_sz[0]]) * self.square_len
        self.board_warped = None

        self.var_terminate = var_terminate
        self.pconverged = np.zeros(self.calib.nintr, dtype=np.bool)

        self.converged = False
        self.tgt_param = None

        # actual user guidance
        self.pose_reached = False
        self.capture = False
        self.still = False
        self.user_info_text = ""

        self.posegen = PoseGeneratorDist(self.img_size)

        # set first pose
        self.set_next_pose()

    def calibrate(self):
        if len(self.calib.keyframes) < 2:
            # need at least 2 keyframes
            return

        pvar_prev = np.diag(self.calib.PCov)[:self.calib.nintr]
        first = len(self.calib.keyframes) == 2

        index_of_dispersion = self.calib.calibrate().copy()

        pvar = np.diag(self.calib.PCov)[:self.calib.nintr]

        if not first:
            total_var_prev = np.sum(pvar_prev)
            total_var = np.sum(pvar)

            if total_var > total_var_prev:
                # del self.calib.keyframes[-1]
                print("note: total var degraded")
                # return

            # check for convergence
            rel_pstd = 1 - np.sqrt(pvar) / np.sqrt(pvar_prev)
            # np.set_printoptions(linewidth=800)
            # print(np.abs(np.sqrt(var) / vals))
            # print(rel_pstd[self.tgt_param])
            #assert rel_pstd[self.tgt_param] >= 0, self.INTRINSICS[self.tgt_param] + " degraded"
            if rel_pstd[self.tgt_param] < 0:
                print(self.INTRINSICS[self.tgt_param] + " degraded")
            for g in self.PARAM_GROUPS:
                if self.tgt_param not in g:
                    continue

                converged = []

                for p in g:
                    # if index_of_dispersion[p] < 0.05:
                    if rel_pstd[p] > 0 and rel_pstd[p] < self.var_terminate:
                        if not self.pconverged[p]:
                            converged.append(self.INTRINSICS[p])
                            self.pconverged[p] = True

                if converged:
                    print("{} converged".format(converged))

        # print(self.calib.get_intrinsics())
        # print(index_of_dispersion)
        index_of_dispersion[self.pconverged] = 0

        self.tgt_param = index_of_dispersion.argmax()

        # how well is the requirement 5x more measurements than unknowns is fulfilled
        # print(self.N_pts*2/self.unknowns, self.N_pts, self.unknowns)
        # print("keyframes: ", len(self.keyframes))

        # print("pvar min", self.pose_var.argmin())
        # print(np.diag(K), cdist)

    def set_next_pose(self):
        nk = len(self.calib.keyframes)

        self.tgt_r, self.tgt_t = self.posegen.get_pose(self.board_units,
                                                       nk,
                                                       self.tgt_param,
                                                       self.calib.K,
                                                       self.calib.cdist)

        self.board.create_maps(self.calib.K, self.calib.cdist, self.img_size)
        self.board_warped = self.board.project(self.tgt_r, self.tgt_t)
        img = self.board_warped
        cv2.imshow('img', img)
        return img

    def pose_close_to_tgt(self):
        if not self.tracker.pose_valid:
            return False

        if self.tgt_r is None:
            return False

        self.overlap[:, :] = self.board_warped[:, :, 1] != 0

        Aa = np.sum(self.overlap)

        tmp = self.board.project(self.tracker.rvec,
                                 self.tracker.tvec,
                                 shadow=True)
        Ab = np.sum(tmp)
        # debug_jaccard(self.overlap, tmp)
        self.overlap *= tmp[:, :]
        Aab = np.sum(self.overlap)

        # circumvents instability during initialization and large variance in depth later on
        jaccard = Aab / (Aa + Ab - Aab)

        return jaccard > 0.85

    def update(self, force=False, dry_run=False):
        """
        @return True if a new pose was captured
        """
        if not self.calib.keyframes and self.tracker.N_pts >= self.allpts // 2:
            # try to estimate intrinsic params from single frame
            self.calib.calibrate([self.tracker.get_calib_pts()])

            if not np.isnan(self.calib.K).any() and self.calib.reperr < self.min_reperr_init:
                self.set_next_pose()  # update target pose
                self.tracker.set_intrinsics(self.calib)
                self.min_reperr_init = self.calib.reperr

        self.pose_reached = force and self.tracker.N_pts > 4

        if self.pose_close_to_tgt():
            self.pose_reached = True

        # we need at least 57.5 points after 2 frames
        # and 15 points per frame from then
        n_required = ((self.calib.nintr + 2 * 6) * 5 + 3) // (2 * 2)  # integer ceil

        if len(self.calib.keyframes) >= 2:
            n_required = 6 // 2 * 5

        self.still = self.tracker.mean_flow is not None and self.tracker.mean_flow < 2
        # use all points instead to ensure we have a stable pose
        self.pose_reached *= self.tracker.N_pts >= n_required

        self.capture = self.pose_reached and (self.still or force)

        if not self.capture:
            return False

        self.calib.keyframes.append(self.tracker.get_calib_pts())

        # update calibration with all keyframe
        self.calibrate()

        # use the updated calibration results for tracking
        self.tracker.set_intrinsics(self.calib)

        self.converged = self.pconverged.all()

        if dry_run:
            # drop last frame again
            del self.caib.keyframes[-1]

        if self.converged:
            self.tgt_r = None
        else:
            self.set_next_pose()

        self._update_user_info()

        return True

    def _update_user_info(self):
        self.user_info_text = ""

        if len(self.calib.keyframes) < 2:
            self.user_info_text = "initialization"
        elif not self.converged:
            action = None
            axis = None
            if self.tgt_param < 2:
                action = "rotate"
                # do not consider r_z as it does not add any information
                axis = self.calib.pose_var[:2].argmin()
            else:
                action = "translate"
                # do not consider t_z
                axis = self.calib.pose_var[3:6].argmin() + 3

            param = self.INTRINSICS[self.tgt_param]
            self.user_info_text = "{} '{}' to minimize '{}'".format(action, self.POSE[axis], param)
        else:
            self.user_info_text = "converged at MSE: {}".format(self.calib.reperr)

        if self.pose_reached and not self.still:
            self.user_info_text += "\nhold camera steady"

    def draw(self, img, mirror=False):
        if self.tgt_r is not None:
            img[self.board_warped != 0] = self.board_warped[self.board_warped != 0]

        if self.tracker.pose_valid:
            self.tracker.draw_axis(img)

        if mirror:
            cv2.flip(img, 1, img)

    def seed(self, imgs):
        for img in imgs:
            self.tracker.detect(img)
            self.update(force=True)

    def write(self, outfile):
        flags = [(cv2.CALIB_FIX_PRINCIPAL_POINT, "+fix_principal_point"),
                 (cv2.CALIB_ZERO_TANGENT_DIST, "+zero_tangent_dist"),
                 (cv2.CALIB_USE_LU, "+use_lu")]

        fs = cv2.FileStorage(outfile, cv2.FILE_STORAGE_WRITE)
        fs.write("calibration_time", datetime.datetime.now().strftime("%c"))
        fs.write("nr_of_frames", len(self.calib.keyframes))
        fs.write("image_width", self.calib.img_size[0])
        fs.write("image_height", self.calib.img_size[1])
        fs.write("board_width", self.tracker.board_sz[0])
        fs.write("board_height", self.tracker.board_sz[1])
        fs.write("square_size", self.square_len)

        flags_str = " ".join([s for f, s in flags if self.calib.flags & f])
        fs.writeComment("flags: " + flags_str)

        fs.write("flags", self.calib.flags)
        fs.write("fisheye_model", 0)
        fs.write("camera_matrix", self.calib.K)
        fs.write("distortion_coefficients", self.calib.cdist)
        fs.write("avg_reprojection_error", self.calib.reperr)
        fs.release()

class UVCVideoCapture:
    def __init__(self, cfg):
        self.manual_focus = True
        self.manual_exposure = True

        imsize = (int(cfg.getNode("image_width").real()), int(cfg.getNode("image_height").real()))

        cam_id = 0
        # TODO XIMEA
        Xi_cam = xiapi.Camera()
        # start communication
        # to open specific device, use:
        # cam.open_device_by_SN('41305651')
        # (open by serial number)
        print('Opening first camera...')
        Xi_cam.open_device()
        Xi_cam.set_exposure(10000)
        print('Exposure was set to %i us' % Xi_cam.get_exposure())

        # create instance of Image to store image data and metadata
        img = xiapi.Image()

        # start data acquisition
        print('Starting data acquisition...')
        Xi_cam.start_acquisition()
        ############### TODO END
        if not cfg.getNode("v4l_id").empty():
            cam_id = "/dev/v4l/by-id/usb-{}-video-index0".format(cfg.getNode("v4l_id").string())

        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, imsize[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imsize[1])
        cap.set(cv2.CAP_PROP_GAIN, 0.0)
        # cap.set(cv2.CAP_PROP_AUTOFOCUS, not self.manual_focus)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        val = 1 / 4 if self.manual_exposure else 3 / 4
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)

        self.cap = cap

        assert self.cap.isOpened()

    def set(self, prop, val):
        self.cap.set(prop, val)

    def read(self):
        return self.cap.read()

def add_camera_controls(win_name, cap):
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)

    if cap.manual_focus:
        focus = 0
        cap.set(cv2.CAP_PROP_FOCUS, focus / 100)
        cv2.createTrackbar("Focus", win_name, focus, 100, lambda v: cap.set(cv2.CAP_PROP_FOCUS, v / 100))

    if cap.manual_exposure:
        exposure = 200
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure / 1000)
        cv2.createTrackbar("Exposure", win_name, exposure, 1000, lambda v: cap.set(cv2.CAP_PROP_EXPOSURE, v / 1000))
