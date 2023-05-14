import pandas as pd
import glob

from helper import preprocess_gwp_asp_cleaning


def main(
    DATAPATH="gwp_data/processed", metafile="meta/columns", OUTPATH="gwp_data/cleaned"
):
    df_meta = pd.read_pickle(metafile)
    source = glob.glob(DATAPATH + "/*.parquet")

    df = pd.DataFrame()
    current_year = 2006
    for i, file in enumerate(sorted(source)):
        new_df = pd.read_parquet(file)

        year = int(str(file).split("/")[-1].split("_")[0])
        if year == current_year:
            # same year: append to df
            df = pd.concat([df, new_df]).copy()
        elif i == len(source) - 1:
            # last file: process df and save
            print(f"Processing year {current_year}...")
            df = preprocess_gwp_asp_cleaning(df, df_meta)
            if len(df) > 0:
                df.to_parquet(OUTPATH + f"/{int(current_year)}.parquet")
        else:
            # new year: process df and save
            print(f"Processing year {current_year}...")
            df = preprocess_gwp_asp_cleaning(df, df_meta)
            if len(df) > 0:
                df.to_parquet(OUTPATH + f"/{int(current_year)}.parquet")

            df = new_df
            current_year = year


if __name__ == "__main__":
    main()
