import json
import re 

def search_entity(sentence):
    # 搜索e1
    e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
    # 搜索e2
    e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]
    sentence = sentence.replace('<e1>' + e1 + '</e1>', ' <e1> ' + e1 + ' </e1> ', 1)
    sentence = sentence.replace('<e2>' + e2 + '</e2>', ' <e2> ' + e2 + ' </e2> ', 1)
    sentence = sentence.strip().split()

    assert '<e1>' in sentence
    assert '<e2>' in sentence
    assert '</e1>' in sentence
    assert '</e2>' in sentence
    return e1, e2, sentence


def convert(path_src, path_tgt):
    with open(path_src, 'r', encoding='utf-8') as fr:
        data = fr.readlines()

    with open(path_tgt, 'w', encoding='utf-8') as fw:
        for i in range(0, len(data), 4):
            id_s, sentence = data[i].strip().split('\t')
            # 去除句子首尾两端的\"
            sentence = sentence[1:-1]
            e1, e2, sentence = search_entity(sentence)
            meta = dict(
                id = id_s,
                relation = data[i+1].strip(),
                head = e1,
                tail = e2,
                sentence = sentence,
                comment = data[i + 2].strip()[8:]
            )
            json.dump(meta, fw, ensure_ascii=False)
            fw.write('\n')

if __name__ == '__main__':
    path_train = './TRAIN_FILE.TXT'
    path_test = './TEST_FILE_FULL.TXT'

    convert(path_train, './train.json')
    convert(path_test, './test.json')