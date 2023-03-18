import sys
sys.path.append('.')

import pandas as pd
import glob
import pyreadstat

import dataprep

def main(DATAPATH="gwp_data/Gallup_World_Poll_091622_ALL_WAVES", metafile="meta/columns", OUTPATH="gwp_data/processed"):
    df_meta = pd.read_pickle("meta/columns")
    source = glob.glob(DATAPATH + "/*.sav")[0]
    for i, (df, meta) in enumerate(pyreadstat.read_file_in_chunks(pyreadstat.read_sav,
                                    file_path=source,
                                    chunksize=50000)):
        gb = df.groupby("YEAR_WAVE")
        for year, group in gb:
            df.columns=meta.column_labels
            df = dataprep.preprocess_gwp_asp(group, meta, df_meta)
            group.to_parquet(OUTPATH + f"/{int(year)}_{i}.parquet")

if __name__ == "__main__":
    main()

