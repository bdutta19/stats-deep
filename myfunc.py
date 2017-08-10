import numpy as np
import cv2
import os
import scipy
from scipy.stats import skew
from scipy import ndimage
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure


def my_weibull(x, l, c, A):
    return A*float(c)/l*(x/float(l))**(c-1)*np.exp(-(x/float(l))**c)


def color_grad_mag(img):
    """
    compute color gradient maps
    img is a GBR image
    """
    img = img[:, :, ::-1]
    img_size, _, _ = img.shape
    temp = img.reshape((img_size * img_size, 3))
    m = np.array([[0.06, 0.63, 0.27], [0.3, 0.04, -0.35], [0.34, -0.6, 0.17]])
    m = m.transpose()
    temp = temp.dot(m)
    temp = temp.reshape((128, 128, 3))
    sobelx0 = cv2.Sobel(temp[:, :, 0], cv2.CV_64F, 1, 0, ksize=5)
    sobely0 = cv2.Sobel(temp[:, :, 0], cv2.CV_64F, 0, 1, ksize=5)
    sobelx1 = cv2.Sobel(temp[:, :, 1], cv2.CV_64F, 1, 0, ksize=5)
    sobely1 = cv2.Sobel(temp[:, :, 1], cv2.CV_64F, 0, 1, ksize=5)
    sobelx2 = cv2.Sobel(temp[:, :, 2], cv2.CV_64F, 1, 0, ksize=5)
    sobely2 = cv2.Sobel(temp[:, :, 2], cv2.CV_64F, 0, 1, ksize=5)
    grad_mag = np.sqrt(sobelx0 * sobelx0 + sobelx1 * sobelx1 + sobelx2 * sobelx2 \
                       + sobely0 * sobely0 + sobely1 * sobely1 + sobely2 * sobely2)
    return grad_mag


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def power_spec(img):
    assert len(img.shape) == 2, 'img should be a grayscale image'
    power = np.fft.fft2(img)
    power = np.abs(power)**2
    return power


def power_spec_param(spec):
    """
    compute parameters of a power magnitude map
    :param spec: power magnitude map
    :return: parameters of a power magnitude map
    """
    img_size, _ = spec.shape
    log_spec_x = np.log10(spec.mean(0))[:img_size / 2]
    log_spec_x = log_spec_x[1:]
    log_spec_y = np.log10(spec.mean(1))[:img_size / 2]
    log_spec_y = log_spec_y[1:]
    freq = np.arange(img_size / 2)
    freq = freq[1:]  # skip zero frequency
    log_freq = np.log10(freq)
    param1, res1, _, _, _ = np.polyfit(log_freq, log_spec_x, 1, full=True)
    param2, res2, _, _, _ = np.polyfit(log_freq, log_spec_y, 1, full=True)
    return param1[0], param1[1], res1[0], param2[0], param2[1], res2[0]


def ave_img(img_root):
    """
    average of images in img_root
    :param img_root: path to the images
    :return: average image
    """
    files = os.listdir(img_root)
    aimg = 0
    it = 1.0
    for imname in files:
        if not imname.endswith('.png'):
            continue
        pass
        img = cv2.imread(os.path.join(img_root, imname))
        aimg = aimg*(it-1)/it + img/it
        it += 1
    return aimg


def luminance_distribution(img_root, output_root=None, tag=None):
    """
    average luminance distribution over images in img_root
    :param img_root: path to the images
    :param output_root: path to write the output values and figures
    :param tag:
    :return: average distribution and its parameters
    """
    files = os.listdir(img_root)
    hist = 0
    it = 1.0
    s = 0
    if output_root is not None:
        assert tag is not None, 'tag must be a string'
        s_list = []
        output_skew = output_root + '/' + tag + '_lum_skew.npz'
        output_name = open(output_root + '/' + tag + '_lum_name.txt', 'w')
    for imname in files:
        if not imname.endswith('.png'):
            continue
        pass
        img = cv2.imread(os.path.join(img_root, imname), 0).astype('float')
        img += 1
        img /= img.mean()
        _hist, b = np.histogram(img.ravel(), 120, [0, 6])
        _s = skew(img.ravel(), bias=False)
        s = s * (it-1)/it + _s/it
        hist = hist * (it-1)/it + _hist/it
        if output_root is not None:
            s_list.append(_s)
            output_name.write(imname + '\n')
        it += 1
    if output_root is not None:
        np.savez(output_skew, np.array(s_list))
        output_name.close()
    return hist, b, s


def contrast_distribution(img_root):
    """
    average contrast distribution of image in img_root
    :param img_root: path to the images
    :return: the histogram of the distribution
    """
    files = os.listdir(img_root)
    hist = 0
    it = 1.0
    for imname in files:
        if not imname.endswith('.png'):
            continue
        pass
        img = cv2.imread(os.path.join(img_root, imname)).astype('float')
        grad_img = color_grad_mag(img)
        _hist, b = np.histogram(grad_img.ravel(), 120, [-80, 8000])
        hist = hist * (it-1)/it + _hist/it
        it += 1
    return hist, b


def random_filter_response_distribution(img_root, kernel, output_root=None, tag=None):
    """
    average random filter response distribution of image in img_root
    :param img_root:
    :param kernel:
    :param output_root:
    :param tag:
    :return:
    """
    files = os.listdir(img_root)
    bins = 0
    k = 0
    if output_root is not None:
        assert tag is not None, 'tag must be a string'
        k_list = []
        output_k = output_root + '/' + tag + '.npz'
        output_name = open(output_root + '/' + tag + '_rdf_name.txt', 'w')
    it = 1.0
    for img_name in files:
        if not img_name.endswith('.png'):
            continue
        pass
        # load image
        img = cv2.imread(img_root + '/' + img_name, 0)
        img = img.astype('float64')
        response = ndimage.convolve(img, kernel, mode='constant', cval=0.0)
        _k = kurtosis(response.ravel())
        # all_std.append(response.std())
        _bins, edges = np.histogram(response, bins=256, range=[-100, 100])
        bins = bins * (it - 1) / it + _bins / it
        k = k * (it - 1) / it + _k / it
        if output_root is not None:
            k_list.append(_k)
            output_name.write(img_name + '\n')
        it += 1
    if output_root is not None:
        np.savez(output_k, np.array(k_list))
        output_name.close()
    return bins, edges, k


def average_power_spectrum(img_root, output_root=None, tag=None):
    """
    average power spectrum of image in img_root
    :param img_root:
    :param output_root:
    :param tag:
    :return:
    """
    files = os.listdir(img_root)
    fimg = 0
    it = 1.0
    if output_root is not None:
        assert tag is not None, 'tag must be a string'
        alpha_x = []
        a_x = []
        res_x = []
        alpha_y = []
        a_y = []
        res_y = []
        output_alpha_x = output_root + '/' + tag + '_alpha_x.npz'
        output_a_x = output_root + '/' + tag + '_a_x.npz'
        output_res_x = output_root + '/' + tag + '_res_x.npz'
        output_alpha_y = output_root + '/' + tag + '_alpha_y.npz'
        output_a_y = output_root + '/' + tag + '_a_y.npz'
        output_res_y = output_root + '/' + tag + '_res_y.npz'
        output_name = open(output_root + '/' + tag + '_spec_name.txt', 'w')
    for imname in files:
        if not imname.endswith('.png'):
            continue
        pass
        img = cv2.imread(os.path.join(img_root, imname), 0)
        _fimg = np.fft.fft2(img)
        _fimg = np.abs(_fimg)**2
        if output_root is not None:
            if (_fimg.mean(0)==0).sum() != 0 or (_fimg.mean(1)==0).sum() != 0:
                continue
            _alpha_x, _a_x, _res_x, _alpha_y, _a_y, _res_y = power_spec_param(_fimg)
            output_name.write(imname + '\n')
            alpha_x.append(_alpha_x)
            a_x.append(_a_x)
            res_x.append(_res_x)
            alpha_y.append(_alpha_y)
            a_y.append(_a_y)
            res_y.append(_res_y)
        fimg = fimg*(it-1)/it + _fimg/it
        it += 1
    if output_root is not None:
        alpha_x = np.array(alpha_x)
        a_x = np.array(a_x)
        res_x = np.array(res_x)
        alpha_y = np.array(alpha_y)
        a_y = np.array(a_y)
        res_y = np.array(res_y)
        np.savez(output_alpha_x, alpha_x)
        np.savez(output_a_x, a_x)
        np.savez(output_res_x, res_x)
        np.savez(output_alpha_y, alpha_y)
        np.savez(output_a_y, a_y)
        np.savez(output_res_y, res_y)
        output_name.close()
    return fimg


def power_spec3d(f_img):
    """
    plot 3d power magnitude map
    :param f_img: power magnitude map
    :return: fig1: 2d power magnitude map, fig2: contour
    """
    # TODO: adjust figure layout
    img_size, _ = f_img.shape
    fig1, _ = plt.subplots()
    ax = Axes3D(fig1)
    x = np.arange(-img_size / 2, img_size / 2, 1)
    y = np.arange(-img_size / 2, img_size / 2, 1)
    x, y = np.meshgrid(x, y)
    ax.plot_surface(x, y, np.log10(np.fft.fftshift(f_img)), rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    ax.set_xlabel('$f_x$', fontsize=18)
    ax.set_ylabel('$f_y$', fontsize=18)
    ax.set_zlabel('||A(f)||', fontsize=18)
    fig1.tight_layout()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.1))
    ax.grid('on')
    return fig1


def weibull_contrast_param(img_root, output_root, tag):
    """
    calculate Weibull parameters
    :param img_root: path to the images
    :return: None
    """
    output_c = output_root + '/' + tag + '_c.npz'
    output_s = output_root + '/' + tag + '_s.npz'
    output_kld = output_root + '/' + tag + '_kld.npz'
    output_name = open(output_root + '/' + tag + '_contrast_name.txt', 'w')
    c_list = []
    s_list = []
    kld_list = []
    files = os.listdir(img_root)
    it = 1.0
    ave_kld = 0.0
    ave_c = 0.0
    ave_mu = 0.0
    ave_s = 0.0
    for imname in files:
        if not imname.endswith('.png'):
            continue
        pass
        # load iamge
        #print ('testing %d of %s ' % (it, tag))
        img = cv2.imread(img_root + '/' + imname[0:-4] + '.png')
        grad_img = color_grad_mag(img)
        data = grad_img.ravel()
        data -= data.min()  # minus minimum edge strength to make mu always be zero
        # evaluate Weibull paratemers
        p0, p1, p2 = scipy.stats.weibull_min.fit(data, floc=0)
        # Kullback-Leibler divergence
        ydata = scipy.stats.weibull_min.pdf(data, p0, p1, p2)
        entropy = scipy.stats.entropy(data, ydata)
        if entropy < 10000:
            # don't take images with too much kld into account
            output_name.write(imname + '\n')
            kld_list.append(entropy)
            c_list.append(p0)
            s_list.append(p2)
            ave_kld = ave_kld * (it - 1) / it + entropy / it
            ave_c = ave_c * (it - 1) / it + p0 / it
            ave_mu = ave_mu * (it - 1) / it + p1 / it
            ave_s = ave_s * (it - 1) / it + p2 / it
            # print('now kld: %.4f, c: %.4f, s: %.4f\n'
            #       % (entropy, p0, p2))
            # print('ave kld: %.4f, c: %.4f, s: %.4f\n'
            #       % (ave_kld, ave_c, ave_s))
            it += 1
    np.savez(output_kld, np.array(kld_list))
    np.savez(output_c, np.array(c_list))
    np.savez(output_s, np.array(s_list))
    output_name.close()


def area_statistic(img_root, output_root, tag):
    """
    compute distribution of areas of connected components,
    as in alvarez1999
    :rtype: numpy.ndarray
    :param img_root: directory of the images
    :return: bins: areas histogram
    """
    k = 16
    img_size = 128
    bins = 0
    files = os.listdir(img_root)
    it = 1.0
    if output_root is not None:
        assert tag is not None, 'tag must be a string'
        param1s = []
        param2s = []
        ress = []
        output_param1 = output_root + '/' + tag + '_param1.npz'
        output_param2 = output_root + '/' + tag + '_param2.npz'
        output_res = output_root + '/' + tag + '_res.npz'
        output_name = open(output_root + '/' + tag + '_area_name.txt', 'w')
    for img_name in files:
        if not img_name.endswith('.png'):
            continue
        pass
        _bins = np.zeros(90)
        # load image
        img = cv2.imread(img_root + '/' + img_name, 0)
        perm = np.sort(img.ravel())
        n_all = perm[range(0, img_size ** 2, img_size ** 2 / (k - 1))]
        n_all0 = np.zeros(n_all.shape)
        n_all0[1:] = n_all[:-1]
        for n, n0 in zip(n_all, n_all0):  # for each lavel
            mask = np.zeros(img.shape)
            mask[img <= n] = 1
            mask[img < n0] = 0
            components = measure.label(mask, background=0)
            for label in np.unique(components):  # for each c.c in this level
                if label != 0:
                    area = (components == label).sum()
                    if area < 90:
                        _bins[area] += 1
        _bins = _bins[1:]
        if output_root is not None:
            s = np.arange(_bins.size)
            s += 1
            st = s[s < 90]
            bt = (_bins+1)[s < 90]
            param, res, _, _, _ = np.polyfit(np.log10(st), np.log10(bt), 1, full=True)
            param1s.append(param[0])
            param2s.append(param[1])
            ress.append(res)
            output_name.write(img_name + '\n')
        bins = bins * (it - 1) / it + _bins / it
    if output_root is not None:
        param1s = np.array(param1s)
        param2s = np.array(param2s)
        ress = np.array(ress)
        np.savez(output_param1, param1s)
        np.savez(output_param2, param2s)
        np.savez(output_res, ress)
        output_name.close()
    return bins
