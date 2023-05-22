import typing as T
from pathlib import Path

import pandas as pd
from sklearn.utils import Bunch

URLS = {
    # Adelie penguin data from: https://doi.org/10.6073/pasta/abc50eed9138b75f54eaada0841b9b86
    "adelie": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.219.3&entityid=002f3893385f710df69eeebe893144ff",
    # Gentoo penguin data from: https://doi.org/10.6073/pasta/2b1cff60f81640f182433d23e68541ce
    "gentoo": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.220.3&entityid=e03b43c924f226486f2f0ab6709d2381",
    # Chinstrap penguin data from: https://doi.org/10.6073/pasta/409c808f8fc9899d02401bdb04580af7
    "chinstrap": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-pal.221.2&entityid=fe853aa8f7a59aa84cdd3197619ef462",
}


DATASET_PATH = Path(__file__).parent / "dataset.csv"


def download_penguins(urls: T.Mapping[str, str] = URLS, path=DATASET_PATH):
    """Download, clean and save the Palmer Penguins dataset

    :params urls: url for the penguins species datasets
    :params path: file path for the .csv file to save results to
    """
    dset_raw = pd.concat([pd.read_csv(url) for url in urls.values()])
    cols = [
        "Species",
        "Culmen Length (mm)",
        "Culmen Depth (mm)",
        "Flipper Length (mm)",
        "Body Mass (g)",
        "Sex",
    ]
    dset = (
        dset_raw[cols]
        .where(lambda x: x["Sex"].isin(["MALE", "FEMALE"]))
        .dropna()
        .pipe(lambda df: pd.get_dummies(df, columns=["Sex"], prefix_sep=" "))
        .drop(columns=["Sex MALE"])
    )
    dset.to_csv(path, index=False)


def load_penguins(path: Path = DATASET_PATH) -> Bunch:
    """Load the Palmer Penguins dataset

    :params path: file path for the .csv file to load
    :returns: features and target data
    """

    dset = pd.read_csv(path)

    features = dset.drop(columns="Species")
    target = dset["Species"].astype("category")

    penguins = Bunch(
        data=features.values,
        feature_names=features.columns.values,
        target=target.cat.codes,
        target_names=target.cat.categories.values,
    )

    return penguins
