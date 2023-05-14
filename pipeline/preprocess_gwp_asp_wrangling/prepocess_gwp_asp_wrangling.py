import glob

from helper import preprocess_gwp_asp_wrangling
import pandas as pd
import pyreadstat


def main(
    DATAPATH="gwp_data/Gallup_World_Poll_091622_ALL_WAVES",
    metafile="meta/columns",
    OUTPATH="gwp_data/processed",
):
    df_meta = pd.read_pickle(metafile)
    source = glob.glob(DATAPATH + "/*.sav")[0]
    for i, (df, meta) in enumerate(
        pyreadstat.read_file_in_chunks(
            pyreadstat.read_sav, file_path=source, chunksize=25000
        )
    ):
        print(f"Processing chunk {i}...")
        gb = df.groupby("YEAR_WAVE")
        for year, group in gb:
            # group.columns = meta.column_labels
            df = preprocess_gwp_asp_wrangling(group, meta, df_meta)
            df.to_parquet(OUTPATH + f"/{int(year)}_{i}.parquet")


if __name__ == "__main__":
    main()
