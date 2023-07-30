IMAGE_DEM = ("./dem/", "dem")
IMAGE_S1 = ("./S1Hand/", "S1Hand")
IMAGE_S2 = ("./S2Hand/", "S2Hand")
IMAGE_LAB = ("./label/", "LabelHand")
DATA_PATH = "../src/downloaded-data/"


class Feature:
    dict = {}

    def __init__(self, name: str):
        self.name = name
        Feature.dict[self.name] = self

    def delete(self):
        del Feature.dict[self.name]

    def get(name: str):
        return Feature.dict[name] or False

    def list():
        return Feature.dict or False

    def access(name: str, dem, s1, s2, lab):
        raise NotImplementedError

    def getName(self):
        return self.name

    def getImage(self):
        raise NotImplementedError

    def getBand(self):
        raise NotImplementedError


class RawFeature(Feature):
    def __init__(self, name: str, image: tuple, band: int):
        super().__init__(name)
        self.image = image
        self.band = band

    def access(self, dem, s1, s2, lab):
        match self.image:
            case "dem":
                return dem
            case "s1":
                return s1[:, self.band]
            case "s2":
                return s2[:, self.band]
            case "lab":
                return lab

    def getImage(self):
        return self.image

    def getBand(self):
        return self.band


class CompositeFeature(Feature):
    def __init__(self, name: str, accessor: callable):
        super().__init__(name)
        self.accessor = accessor

    def access(self, dem, s1, s2, lab):
        return self.accessor(Feature.get)


RawFeature(name="DEM", image=IMAGE_DEM, band=0)
RawFeature(name="SAR_VV", image=IMAGE_S1, band=0)
RawFeature(name="SAR_VH", image=IMAGE_S1, band=1)
RawFeature(name="OPT_R", image=IMAGE_S2, band=3)
RawFeature(name="OPT_G", image=IMAGE_S2, band=3)
RawFeature(name="OPT_B", image=IMAGE_S2, band=2)
RawFeature(name="OPT_N", image=IMAGE_S2, band=7)
RawFeature(name="OPT_RE1", image=IMAGE_S2, band=4)
RawFeature(name="OPT_RE2", image=IMAGE_S2, band=5)
RawFeature(name="OPT_RE3", image=IMAGE_S2, band=6)
RawFeature(name="OPT_NNIR", image=IMAGE_S2, band=8)
RawFeature(name="OPT_SWIR1", image=IMAGE_S2, band=11)
RawFeature(name="OPT_SWIR2", image=IMAGE_S2, band=12)
RawFeature(name="QC", image=IMAGE_LAB, band=0)

CompositeFeature(
    name="NDWI",
    accessor=lambda r: (r["OPT_G"] - r["OPT_N"]) / (r["OPT_G"] + r["OPT_N"]),
)
CompositeFeature(
    name="AWEI",
    accessor=lambda r: 4 * (r["OPT_G"] - r["OPT_SWIR1"])
    - 0.25 * (r["OPT_N"] + 11 * r["OPT_SWIR2"]),
)
CompositeFeature(
    name="AEWISH",
    accessor=lambda r: r["OPT_B"]
    + (5 / 2) * r["OPT_G"]
    - 1.5 * (r["OPT_N"] + r["OPT_SWIR1"])
    - r["OPT_SWIR2"] / 4,
)
CompositeFeature(
    name="MNDWI",
    accessor=lambda r: (r["OPT_G"] - r["OPT_SWIR1"]) / (r["OPT_G"] + r["OPT_SWIR1"]),
)


print(
    f"""Testing features
      DEM: {Feature.get("DEM").getName()}
      MNDWI: {Feature.get("MNDWI").getName()}
      
      """
)
