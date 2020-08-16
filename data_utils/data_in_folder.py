from shutil import copyfile, copy
from mat_to_csv import mat_to_csv

imdb_df = mat_to_csv().convert(path_mat="dataset/imdb.mat", key="imdb")
# wiki_df = mat_to_csv().convert(path_mat="dataset/wiki.mat", key="wiki")

imdb_df.head()

# for index, row in imdb_df.tail(imdb_df.shape[0]-263969).iterrows():
#     sys.stdout.write("\r%d%%" % ((index/imdb_df.shape[0])*100))
#     sys.stdout.write(row.full_name)
#     sys.stdout.flush()
#     dir = 'dataset/imdb/' + str(row.full_name)
#     if not os.path.exists(dir): os.makedirs(dir)
    
#     try:
#         copy('D:/Datasets/imdb_crop/' + row.full_path, dir)
#     except Exception as e:
#         print(e)