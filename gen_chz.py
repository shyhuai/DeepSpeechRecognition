
fn = 'data/thchs_test.txt'
target = 'data/thchs_pure.txt'
tf = open(target, 'w')
with open(fn, 'r') as f:
    for l in f.readlines():
        zh = l.split(' ')[-1]
        tf.write(zh)
tf.close()
