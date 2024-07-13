from Map import Map
from Reduce import Reduce

# 执行Map过程
def run_maps(root_dir='../source_data', key_words_file='../words.txt'):
    # 读取key_word列表
    key_words_file = open(key_words_file, 'r')
    key_words = key_words_file.read().splitlines()
    key_words_file.close()

    # 将key_word列表转换为字典，后续可加快比对速度
    key_words = {key_word: 1 for key_word in key_words}

    # 创建9个Map节点
    # 采用多线程模拟分布式节点
    map_nodes = list()
    for i in range(9):
        map_node = Map(id=i + 1, root_dir=f'{root_dir}/folder_{i + 1}', key_words=key_words)
        map_nodes.append(map_node)
        map_node.start()
    for map_node in map_nodes:
        map_node.join()

    return map_nodes

# 执行Reduce过程
def run_reduces(map_nodes: list):
    # 创建3个Reduce节点
    # 采用多线程模拟分布式节点
    reduce_nodes = list()

    # Shuffle操作：将Map节点尽可能平均分配到3个Reduce上
    for i in range(3):
        inputs = list()
        for j in range(9):
            inputs.extend(map_nodes[j].output[i])
        reduce_node = Reduce(id=i+1, inputs=inputs)
        reduce_nodes.append(reduce_node)
        reduce_node.start()
    for reduce_node in reduce_nodes:
        reduce_node.join()

    return reduce_nodes

# 由9个Reduce节点合并得到前1000项单词
def get_sorted_words(reduce_nodes: list):
    words = dict()
    for reduce_node in reduce_nodes:
        output = reduce_node.output
        for word in output.keys():
            if words.get(word) is None:
                words[word] = {'count': 0, 'relations': list()}
            words[word]['count'] += output[word]['count']
            words[word]['relations'].extend(output[word]['relations'])

    sorted_words = sorted(words.items(), key=lambda word: word[1]['count'], reverse=True)[:1000]
    sorted_words_dict = {sorted_words[i][0]: i for i in range(len(sorted_words))}

    print('Write words starts!')
    with open('outputs/words_count.txt', 'w+', encoding='utf-8') as f:
        for word in sorted_words:
            f.write(f"{word[0]} : {word[1]['count']}\n")
    with open('outputs/title_to_keys.txt', 'w+', encoding='utf-8') as f:
        for word in sorted_words:
            relations = list()
            for relation_word in word[1]['relations']:
                if sorted_words_dict.get(relation_word) is not None:
                    relations.append(relation_word)
            word[1]['relations'] = relations
            f.write(f"{word[0]} -> {word[1]['relations']}\n")
    with open('outputs/title_to_keys_raw.txt', 'w+', encoding='utf-8') as f:
        sorted_words = [(word[0], word[1]['relations']) for word in sorted_words]
        f.write(str(sorted_words))
    print('Write words done!')

    return sorted_words

if __name__ == '__main__':
    map_nodes = run_maps()
    reduce_nodes = run_reduces(map_nodes)
    sorted_words = get_sorted_words(reduce_nodes)
    # sorted_words = load_sorted_words()








