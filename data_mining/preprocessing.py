from datetime import datetime
import pandas
import time

# constant
FILE_NAMES = ['account.csv', 'card.csv', 'client.csv',
              'disp.csv', 'district.csv', 'loan.csv', 'order.csv', 'trans.csv']
FILE_NON_VALUE_THRESH = [4, 4, 3, 4, 8, 8, 6, 8]
PRIMARY_KEY = ['account_id', 'card_id', 'client_id', 'disp_id', 'district_id', 'loan_id', 'order_id', 'trans_id']
FOREIGN_KEY = [['district_id'], ['disp_id'], ['district_id'], ['client_id', 'account_id'], [], ['account_id'],
               ['account_id'], ['account_id']]

TYPE_INFO = [{'account_id': 'int', 'district_id': 'int', 'date': 'date', 'frequency': 'str'},
             {'card_id': 'int', 'disp_id': 'int', 'type': 'str', 'issued': 'date'},
             {'client_id': 'int', 'birth_number': 'int', 'district_id': 'int'},
             {'disp_id': 'int', 'client_id': 'int', 'account_id': 'int', 'type': 'str'},
             {'district_id': 'int', 'district_name': 'str', 'region': 'str', 'hab_number': 'int', 'city_number': 'int',
              'ave_salary': 'int', 'umemploy_rate': 'float', 'crime_number': 'int'},
             {'loan_id': 'int', 'account_id': 'int', 'date': 'date', 'amount': 'int', 'duration': 'int',
              'payments': 'int', 'status': 'str', 'payduration': 'int'},
             {'order_id': 'int', 'account_id': 'int', 'bank_to': 'str', 'account_to': 'int', 'amount': 'int',
              'k_symbol': 'str'},
             {'trans_id': 'int', 'account_id': 'int', 'date': 'datetime', 'type': 'str', 'operation': 'str',
              'amount': 'int', 'balance': 'int', 'k_symbol': 'str', 'bank': 'str', 'account': 'int'}]

special_filter_switch = {
    'account.csv': lambda x: account_filter(x),
    'card.csv': lambda x: card_filter(x),
    'client.csv': lambda x: client_filter(x),
    'disp.csv': lambda x: disp_filter(x),
    'loan.csv': lambda x: loan_filter(x),
    'district.csv': lambda x: x,
    'order.csv': lambda x: order_filter(x),
    'trans.csv': lambda x: trans_filter(x)
}

# global variable
primary_key_values = dict()
account_owner_type = set()


# cache primary key
def store_primary_key(data, primary_key):
    global primary_key_values
    ids = set()
    for key in data[primary_key]:
        ids.add(int(key))
    primary_key_values[primary_key] = ids


# remove duplicate
def remove_duplicate(data, primary_key):
    print('before remove duplicate: ', len(data))
    res = data.drop_duplicates()
    res = res.drop_duplicates(subset=primary_key)
    print('after remove duplicate: ', len(res))
    return res


# check type is confirm
def check_type(data, type_dict):
    type_switch = {
        'int': lambda x: int(x),
        'float': lambda x: float(x),
        'str': lambda x: str(x),
        'date': lambda x: time.strptime(x, "%Y/%m/%d %H:%M"),
        'datetime': lambda x: time.strptime(x, "%Y-%m-%d %H:%M:%S")
    }
    error_line = []
    for i in data.index.values:
        for name in type_dict:
            col_type = type_dict[name]
            try:
                type_switch[col_type](data[name][i])
            except ValueError:
                error_line.append(i)
                print('drop lines name:', name, 'index:', i, 'type:', col_type, 'value:', data[name][i])
    print("type check drop ", len(error_line), "lines of data")
    res = data.drop(error_line)
    return res


def remove_null(data, thresh):
    print('before remove null: ', len(data))
    res = data.dropna(thresh=thresh)
    print('after remove null: ', len(res))

    return res


# check foreign key integrity
def check_integrity_for_key(data, foreign_key_list):
    error_line = []
    for foreign_key in foreign_key_list:
        primary_key = primary_key_values[foreign_key]
        for i in data.index.values:
            val = int(data[foreign_key][i])
            if val not in primary_key:
                error_line.append(i)
                print('foreign key name: ', foreign_key, 'value:', val,
                      ' not correspond to primary key')

    print('integrity check drop ', len(error_line), 'lines of data')
    res = data.drop(error_line)
    return res


# for account.csv
def account_filter(data):
    frequency_set = {'POPLATEK MESICNE', 'POPLATEK TYDNE', 'POPLATEK PO OBRATU'}
    error_line = []
    for i in data.index.values:
        if data['frequency'][i] not in frequency_set:
            error_line.append(i)
            print('drop lines name: frequency', 'index:', i, 'value:', data['frequency'][i])
    print('account filter drop ', len(error_line), 'lines of data')
    res = data.drop(error_line)
    return res


# for card.csv
def card_filter(data):
    type_set = {'junior', 'classic', 'gold'}
    error_line = []
    for i in data.index.values:
        if data['type'][i] not in type_set:
            error_line.append(i)
            print('drop lines name: type', 'index:', i, 'value:', data['type'][i])
    print('card filter drop', len(error_line), 'lines of data')
    res = data.drop(error_line)
    return res


# for client.csv
def client_filter(data):
    error_line = []
    gender = []
    age = []
    for i in data.index.values:
        birth = str(data['birth_number'][i])
        is_female = False
        if len(birth) == 5:
            birth = birth[0:4] + '0' + birth[4]

        birth = '19' + birth

        if len(birth) != 8:
            error_line.append(i)
            print('drop lines name: birth', 'index:', i, 'value:', birth)
        elif int(birth[4:6]) > 50:
            val = int(birth[4:6])
            val -= 50
            birth = birth[0:4] + str(val) + birth[6:8]
            is_female = True
        try:
            # check date is valid
            cur_age = int((datetime.strptime('2000-01-01', '%Y-%m-%d') - datetime.strptime(birth, "%Y%m%d")).days / 365)
            age.append(cur_age)
        except ValueError:
            error_line.append(i)
            print('drop lines name: birth', 'index:', i, 'value:', birth)
            continue

        if is_female:
            gender.append('F')
        else:
            gender.append('M')

    print('client fiter drop', len(error_line), 'lines of data')
    res = data.drop(error_line)
    res['gender'] = gender
    res['age'] = age
    return res


# for disp.csv
def disp_filter(data):
    global account_owner_type
    type_set = {'OWNER', 'DISPONENT'}
    error_line = []
    for i in data.index.values:
        if data['type'][i] not in type_set:
            error_line.append(i)
            print('drop lines name: type', 'index:', i, 'value:', data['type'][i])
        else:
            # for loan check
            if data['type'][i] == 'OWNER':
                account_owner_type.add(int(data['account_id'][i]))
    print('disp filter drop', len(error_line), 'lines of data')
    res = data.drop(error_line)
    return res


# for order.csv
def order_filter(data):
    k_symbol_set = {'POJISTNE', 'SIPO', 'LEASING', 'UVER', ' '}
    error_line = []
    for i in data.index.values:
        if data['k_symbol'][i] not in k_symbol_set:
            error_line.append(i)
            print('drop lines name: k_symbol', 'index:', i, 'value:', data['k_symbol'][i])
    print('order filter drop', len(error_line), 'lines of data')
    res = data.drop(error_line)
    return res


# for trans.csv
def trans_filter(data):
    type_set = {'PRIJEM', 'VYDAJ'}
    error_line = []
    for i in data.index.values:
        if data['type'][i] not in type_set:
            error_line.append(i)
            print('drop lines name: type', 'index:', i, 'value:', data['type'][i])
    print('trans filter drop', len(error_line), 'lines of data')
    res = data.drop(error_line)
    return res


# for loan filter
def loan_filter(data):
    status_set = {'A', 'B', 'C', 'D'}
    error_line = []
    for i in data.index.values:
        # check status
        if data['status'][i] not in status_set:
            error_line.append(i)
            print('drop lines name: status', 'index:', i, 'value:', data['status'][i])
        # check duration
        elif data['duration'][i] % 12 != 0:
            error_line.append(i)
            print('drop lines name: duration', 'index:', i, 'value:', data['duration'][i])
        # check amount
        elif data['duration'][i] * data['payments'][i] != data['amount'][i]:
            error_line.append(i)
            print('drop lines name: amount', 'index:', i, 'value:', data['amount'][i])
        # only owner account can lend money
        elif data['account_id'][i] not in account_owner_type:
            error_line.append(i)
            print('drop lines name: account_id', 'index:', i, 'value:', data['account_id'][i])
    print('loan filter drop', len(error_line), 'lines of data')
    res = data.drop(error_line)
    return res


def special_filter_router(name, data):
    return special_filter_switch[name](data)


if __name__ == '__main__':
    raw_data_list = []
    # load file
    for i in range(8):
        file = FILE_NAMES[i]
        file = 'dataset/' + file
        raw_data_list.append(pandas.read_csv(file))

    # general cleaning work
    for i in range(8):
        raw_data = raw_data_list[i]
        file = FILE_NAMES[i]
        print("general cleaning work begin for file: " + file)
        raw_data = remove_null(raw_data, FILE_NON_VALUE_THRESH[i])
        raw_data = remove_duplicate(raw_data, PRIMARY_KEY[i])
        raw_data = check_type(raw_data, TYPE_INFO[i])
        store_primary_key(raw_data, PRIMARY_KEY[i])
        raw_data_list[i] = raw_data
        print('---------------------------------------------')

    # can only begin when primary key is stored
    for i in range(8):
        raw_data = raw_data_list[i]
        file = FILE_NAMES[i]
        print("check integrity begin for file: " + file)
        raw_data = check_integrity_for_key(raw_data, FOREIGN_KEY[i])
        raw_data_list[i] = raw_data
        print('---------------------------------------------')

    # special cleaning work
    for i in range(8):
        raw_data = raw_data_list[i]
        file = FILE_NAMES[i]
        print("special filter begin for file: " + file)
        raw_data = special_filter_router(file, raw_data)
        raw_data_list[i] = raw_data
        print('---------------------------------------------')

    # write preprocessed dataset
    for i in range(8):
        raw_data = raw_data_list[i]
        file = FILE_NAMES[i]
        raw_data.to_csv('preprocessed_dataset/' + file, sep=',', header=True, index=False)
