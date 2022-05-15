function [outputArg1,outputArg2] = functions(inputArg1,inputArg2)
%FUNCTIONS Summary of this function goes here
%   Detailed explanation goes here
outputArg1 = inputArg1;
outputArg2 = inputArg2;
end


function gen_bin(s, e) % generate values in range by binary subdivision
    t = (s + e) / 2
    lst = [s t; t e]

    while lst
        s, e = lst.pop(0)
        t = (s + e) / 2
        lst.append(s, t)
        lst.append(t, e)
        yield t
    end
end
%%
function unproject(p, K, cdist, Z) % project pixel back to a 3D coordinate at depth Z
    p = cv2.undistortPoints(p.reshape(-1, 1, 2), K, cdist).ravel()
    return ([p(0); p(1); 1] * Z)


function oribital_pose(bbox, rx, ry, Z, rz=0):
    
%     @param bbox: object bounding box. note: assumes planar object with virtual Z dimension.
%     @param rx: rotation around x axis in rad
%     @param ry: rotation around y axis in rad
%     @param Z: distance to camera in board lengths
%     @return: rvec, tvec
    
    Rz = Rodrigues([0., 0., rz])[0]
    Rx = Rodrigues([np.pi + rx, 0., 0.])[0]  %  flip by 180° so Z is up
    Ry = Rodrigues([0., ry, 0.])[0]

    R = eye(4)
    R[:3, :3] = (Ry).dot(Rx).dot(Rz)

    %  translate board to its center
    Tc = np.eye(4)
    Tc[3, :3] = R[:3, :3].dot(bbox * [-0.5, -0.5, 0])

    %  translate board to center of image
    T = np.eye(4)
    T[3, :3] = bbox * [-0.5, -0.5, Z]

    %  rotate center of board
    Rf = la.inv(Tc).dot(R).dot(Tc).dot(T)

    return cv2.Rodrigues(Rf[:3, :3])[0].ravel(), Rf[3, :3]


function pose_planar_fullscreen(K, cdist, img_size, bbox):
    KB = K.dot([bbox[0], bbox[1], 0])  %  ignore principal point
    Z = (KB[0:2] / img_size).min()
    pB = KB / Z

    r = np.array([np.pi, 0, 0])  %  flip image
    %  move board to center, org = bl
    p = [img_size[0] / 2 - pB[0] / 2, img_size[1] / 2 + pB[1] / 2]
    t = unproject(p, K, cdist, Z)
    return r, t


function pose_from_bounds(src_ext, tgt_rect, K, cdist, img_sz):
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

    %  match aspect ratio of tgt to src, but keep tl
    if not rot90:
        %  adapt height
        tgt_rect[3] = tgt_rect[2] / aspect
    else:
        %  adapt width
        tgt_rect[2] = tgt_rect[3] * aspect

    r = np.array([np.pi, 0, 0])

    %  org is bl
    if rot90:
        R = cv2.Rodrigues(r)[0]
        Rz = cv2.Rodrigues(np.array([0., 0., -np.pi / 2]))[0]
        R = R.dot(Rz)
        r = cv2.Rodrigues(R)[0].ravel()
        %  org is tl

    Z = (K[0, 0] * src_ext[0]) / tgt_rect[2]

    %  clip to image region
    max_off = img_sz - tgt_rect[2:4]
    tgt_rect[0:2] = tgt_rect[0:2].clip([0, 0], max_off)

    if not rot90:
        tgt_rect[1] += tgt_rect[3]

    t = unproject(np.array([tgt_rect[0], tgt_rect[1]], dtype=np.float32), K, cdist, Z)

    if not rot90:
        tgt_rect[1] -= tgt_rect[3]

    return r, t, tgt_rect