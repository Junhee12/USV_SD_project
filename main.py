# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(test):
    print('test a : [%d, %d] ' % (test['a'][0], test['a'][1]))


def update(test):
    test['a'][0] += 1

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    test = {'a':[1,1], 'b':[2,2]}

    update(test)
    update(test)
    update(test)
    update(test)

    print_hi(test)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
