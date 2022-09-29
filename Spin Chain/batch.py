import os

decay = 1

update_times = 10

pnt = 32

list = [(2, 'b0602', decay, update_times),(4, 'b0604', decay, update_times),(8, 'b0608', decay, update_times),(16, 'b0616', decay, update_times),(32, 'b0632', decay, update_times),(64, 'b0664', decay, update_times)]


def main():
    for object in list:
        os.system('python Data_Gen.py %s %s %s %s' %(object[0], object[1], object[2], object[3]))

if __name__ == "__main__":
    main()