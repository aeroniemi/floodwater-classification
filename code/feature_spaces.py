IMAGE_DEM = ("./dem/", "dem", "dem")
IMAGE_S1 = ("./S1Hand/", "S1Hand", "s1")
IMAGE_S2 = ("./S2Hand/", "S2Hand", "s2")
IMAGE_LAB = ("./label/", "LabelHand", "lab")
DATA_PATH = "../src/downloaded-data/"

import numpy as np
from skimage.color import rgb2hsv
from skimage.transform import rescale, resize
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# import richdem as rd
class Feature:
    dict = {}

    def __init__(self, name: str, local_rescale=True):
        self.name = name
        self.local_rescale = local_rescale
        Feature.dict[self.name] = self

    def __str__(self):
        return f"{self.__class__.__name__}: {self.getName()}"

    def __repr__(self):
        return self.__str__()

    def delete(self):
        del Feature.dict[self.name]

    def get(name: str):
        return Feature.dict[name] or False

    def getMany(features):
        return list(map(Feature.get, features))

    def list():
        return Feature.dict or False

    def prepare(dem, s1, s2, lab):
        raise NotImplementedError
    def prepare_unscaled(dem, s1, s2, lab):
        raise NotImplementedError

    def access(self, dem, s1, s2, lab):
        # print(dem.shape)
        layer = self.prepare(dem, s1, s2, lab)
        # print(f"15: {layer.shape}")
        return layer
    def access_unscaled(self, dem, s1, s2, lab):
        # print(dem.shape)
        layer = self.prepare_unscaled(dem, s1, s2, lab)
        # print(layer, self)
        if self.local_rescale and layer is np.ndarray and (len(layer.shape) == 2 or layer.shape[2]==1):
            # print("scaling")
            layer = np.reshape(MinMaxScaler().fit_transform(np.concatenate(layer)), layer.shape)
        # print(f"15: {layer.shape}")
        return layer

    def getName(self):
        return self.name

    def getImage(self):
        raise NotImplementedError

    def getBand(self):
        raise NotImplementedError

    def getValues(dem, s1, s2, lab):
        def miniGet(name):
            feature = Feature.get(name)
            return feature.access(dem, s1, s2, lab)

        return miniGet
    def getValues_unscaled(dem, s1, s2, lab):
        def miniGet(name):
            feature = Feature.get(name)
            return feature.access_unscaled(dem, s1, s2, lab)

        return miniGet


class RawFeature(Feature):
    def __init__(self, name: str, image: tuple, band: int):
        super().__init__(name)
        self.image = image
        self.band = band

    def prepare(self, dem, s1, s2, lab):
        match self.image[2]:
            case "dem":
                return dem[:, :, 0]
            case "s1":
                return s1[:, :, self.band]
            case "s2":
                return s2[:, :, self.band]
            case "lab":
                return lab[:, :, 0]
    def prepare_unscaled(self, dem, s1, s2, lab):
        return self.prepare(dem, s1, s2, lab)
    def getImage(self):
        return self.image

    def getBand(self):
        return self.band


class CompositeFeature(Feature):
    def __init__(self, name: str, accessor: callable=None, unscaled_accessor: callable=None):
        super().__init__(name)
        self.accessor = accessor
        self.unscaled_accessor = unscaled_accessor

    def prepare(self, dem, s1, s2, lab):
        if self.accessor is not None:
            return self.accessor(Feature.getValues(dem, s1, s2, lab))
        return None
    def prepare_unscaled(self, dem, s1, s2, lab):
        if self.unscaled_accessor is not None:
            # print(self, "valid accessor")
            return self.unscaled_accessor(Feature.getValues_unscaled(dem, s1, s2, lab))
        return None


RawFeature(name="DEM", image=IMAGE_DEM, band=0)

RawFeature(name="SAR_VV", image=IMAGE_S1, band=0)
RawFeature(name="SAR_VH", image=IMAGE_S1, band=1)

RawFeature(name="OPT_B", image=IMAGE_S2, band=1)
RawFeature(name="OPT_G", image=IMAGE_S2, band=2)
RawFeature(name="OPT_R", image=IMAGE_S2, band=3)
RawFeature(name="OPT_RE1", image=IMAGE_S2, band=4)
RawFeature(name="OPT_RE2", image=IMAGE_S2, band=5)
RawFeature(name="OPT_RE3", image=IMAGE_S2, band=6)
RawFeature(name="OPT_N", image=IMAGE_S2, band=7)
RawFeature(name="OPT_NNIR", image=IMAGE_S2, band=8)
RawFeature(name="OPT_SWIR1", image=IMAGE_S2, band=11)
RawFeature(name="OPT_SWIR2", image=IMAGE_S2, band=12)
RawFeature(name="QC", image=IMAGE_LAB, band=0)

CompositeFeature(
    name="compositeTest",
    accessor=lambda r: print(*r("OPT_G")),
)
CompositeFeature(
    name="NDWI",
    accessor=lambda r: (r("OPT_G") - r("OPT_N"))
    + 1e-20 / (r("OPT_G") + r("OPT_N") + 1e-20),
)


def hsvrgb(r):
    hsv = rgb2hsv(np.array(r("RGB")).T)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


def hsvo3(r):
    hsv = rgb2hsv(np.array(r("O3")).T)
    return hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]


CompositeFeature(
    name="HSV(RGB)",
    accessor=hsvrgb,
)
CompositeFeature(name="RGB", accessor=lambda r: (r("OPT_R"), r("OPT_G"), r("OPT_B")))
CompositeFeature(name="O3", accessor=lambda r: (r("OPT_SWIR2"), r("OPT_N"), r("OPT_R")))
CompositeFeature(
    name="HSV(O3)",
    accessor=hsvo3,
)
CompositeFeature(
    name="AWEI",
    accessor=lambda r: 4 * (r("OPT_G") - r("OPT_SWIR1"))
    - 0.25 * (r("OPT_N") + 11 * r("OPT_SWIR2")),
)
CompositeFeature(
    name="AWEISH",
    accessor=lambda r: r("OPT_B")
    + (5 / 2) * r("OPT_G")
    - 1.5 * (r("OPT_N") + r("OPT_SWIR1"))
    - r("OPT_SWIR2") / 4,
)
CompositeFeature(
    name="MNDWI",
    accessor=lambda r: (r("OPT_G") - r("OPT_SWIR1") + 1e-20)
    / (r("OPT_G") + r("OPT_SWIR1") + 1e-20),
)
CompositeFeature(
    name="SAVI",
    accessor=lambda r: ((r("OPT_N") - r("OPT_R"))
    / (r("OPT_N") + r("OPT_R") + 0.5)*(1+0.5)),
)
CompositeFeature(
    name="WRI",
    accessor=lambda r: ((r("OPT_G") + r("OPT_R"))
    / (r("OPT_N") + r("OPT_SWIR1"))),
)
CompositeFeature(
    name="SARR",
    accessor=lambda r: (r("SAR"), r("SAR_VV")+0.000001/r("SAR_VH")+0.000001)
)

CompositeFeature(name="cAWEI", accessor=lambda r: (r("AWEI"), r("AWEISH")))
CompositeFeature(name="cAWEI+SAVI+WRI", accessor=lambda r: (r("cAWEI"), r("SAVI"), r("WRI")))
CompositeFeature(name="cAWEI+SAVI", accessor=lambda r: (r("cAWEI"), r("SAVI")))
CompositeFeature(name="cAWEI+WRI", accessor=lambda r: (r("cAWEI"), r("WRI")))
CompositeFeature(name="SAR_cAWEI+SAVI", accessor=lambda r: (r("SAR"),r("cAWEI"), r("SAVI")))
CompositeFeature(name="SARR_SAVI", accessor=lambda r: (r("SARR"),r("SAVI")))
CompositeFeature(name="SARR_SAVI+WRI", accessor=lambda r: (r("SARR_SAVI"),r("WRI")))
CompositeFeature(name="cNDWI", accessor=lambda r: (r("NDWI"), r("MNDWI")))
CompositeFeature(name="SAR", accessor=lambda r: (r("SAR_VV"), r("SAR_VH")))
CompositeFeature(
    name="OPT",
    accessor=lambda r: (
        r("OPT_R"),
        r("OPT_G"),
        r("OPT_B"),
        r("OPT_N"),
        r("OPT_RE1"),
        r("OPT_RE2"),
        r("OPT_RE3"),
        r("OPT_NNIR"),
        r("OPT_SWIR1"),
        r("OPT_SWIR2"),
    ),
)

CompositeFeature(name="RGBN", accessor=lambda r: (r("RGB"), r("OPT_N")))
CompositeFeature(name="cAWEI+cNDWI", accessor=lambda r: (r("cAWEI"), r("cNDWI")))
CompositeFeature(
    name="HSV(O3)+cAWEI+cNDWI",
    accessor=lambda r: (r("HSV(O3)"), r("cAWEI"), r("cNDWI")),
)
CompositeFeature(name="SAR_OPT", accessor=lambda r: (r("SAR"), r("OPT")))
CompositeFeature(name="SAR_O3", accessor=lambda r: (r("SAR"), r("O3")))
CompositeFeature(name="SAR_RGB", accessor=lambda r: (r("SAR"), r("RGB")))
CompositeFeature(name="SAR_RGBN", accessor=lambda r: (r("SAR"), r("RGBN")))
CompositeFeature(name="SAR_HSV(RGB)", accessor=lambda r: (r("SAR"), r("HSV(RGB)")))
CompositeFeature(name="SAR_HSV(O3)", accessor=lambda r: (r("SAR"), r("HSV(O3)")))
CompositeFeature(name="SAR_cNDWI", accessor=lambda r: (r("SAR"), r("cNDWI")))
CompositeFeature(name="SAR_cAWEI", accessor=lambda r: (r("SAR"), r("cAWEI")))
CompositeFeature(
    name="SAR_cAWEI+cNDWI", accessor=lambda r: (r("SAR"), r("cAWEI"), r("cNDWI"))
)
CompositeFeature(
    name="SAR_HSV(O3)+cAWEI+cNDWI",
    accessor=lambda r: (r("SAR"), r("HSV(O3)"), r("cAWEI+cNDWI")),
)
CompositeFeature(
    name="DEM_SAR_HSV(O3)+cAWEI+cNDWI",
    accessor=lambda r: (r("DEM"), r("SAR_HSV(O3)+cAWEI+cNDWI")),
)
CompositeFeature(
    name="LDEM_SAR_HSV(O3)+cAWEI+cNDWI",
    accessor=lambda r: (r("LDEM"), r("SAR_HSV(O3)+cAWEI+cNDWI")),
)
CompositeFeature(
    name="DEM_SAR",
    accessor=lambda r: (r("DEM"), r("SAR")),
)
CompositeFeature(
    name="DP_SAR",
    accessor=lambda r: r("SAR"),
    unscaled_accessor=lambda r: r("DP")
)
CompositeFeature(
    name="SDEM+LDEM_SAR",
    accessor=lambda r: (r("SDEM"), r("LDEM"), r("SAR")),
)
CompositeFeature(
    name="ACU+SDEM+LDEM_SAR",
    accessor=lambda r: r("SAR"),
    unscaled_accessor=lambda r: (r("ACU"), r("SDEM"), r("LDEM")),
)
CompositeFeature(
    name="LDEM",
    unscaled_accessor=lambda r: r("DEM"),
)

def slope(r):
    dem = rd.rdarray(r('DEM'), no_data=-9999)
    rd.FillDepressions(dem, epsilon=True, in_place=True)

    slope = rd.TerrainAttribute(dem, attrib='slope_riserun')
    return np.array(slope)
def accumulation(r):
    dem = rd.rdarray(r('DEM'), no_data=-9999)
    rd.FillDepressions(dem, epsilon=True, in_place=True)
    accum_d8 = rd.FlowAccumulation(dem, method='D8')
    return np.array(accum_d8)
2
def depression(r):
    dem = rd.rdarray(getlowresdem(r), no_data=-9999)
    filled = rd.FillDepressions(dem, epsilon=True, in_place=False)
    res = (filled-dem) > 0
    return upscale(np.array(res))

def getlowresdem(r):
    return rescale(r('DEM'), 1/3)
def upscale(feature):
    return resize(feature, (512, 512))

CompositeFeature(
    name="SDEM",
    unscaled_accessor=slope
)
CompositeFeature(
    name="ACU",
    unscaled_accessor=accumulation
)
CompositeFeature(
    name="DP",
    accessor = lambda r: r("cAWEI"),
    unscaled_accessor=depression
)
CompositeFeature(
    name="DP_SAR_HSV(O3)+cAWEI+cNDWI",
    accessor=lambda r:  r("SAR_HSV(O3)+cAWEI+cNDWI"),
    unscaled_accessor=depression
)