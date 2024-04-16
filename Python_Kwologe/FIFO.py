class Queue:
    def __init__(self):
        self.queue = []  # Inicializa a clase Queue com uma array vazia

    def enqueue(self, element):
        self.queue.append(element)  # Adicionar um elento na fila;

    def dequeue(self):
        if self.is_empty():  # Verificar se 
                             # a fila está vazia, se estiver retorna 
                             # apenas um retorna um print 
                             
            print("Queue is empty")
            return None
        else:
            # caso exista dados na fila  
            # remove o primeiro elemento da fila
            return self.queue.pop(0)  # Remove and return the first element

    def is_empty(self):
        return len(self.queue) == 0  # Returne se estiver vazia, caso não esteja vazia retorna false

    def size(self):
        return len(self.queue)  # Returne o tamanho do array/queue

    def print_queue(self):
        print(self.queue)  # Printa a fila

# Cria a instancia do objeto Queue
q = Queue()

# Adicionar elementos na queue
q.enqueue(1)
q.enqueue(2)
q.enqueue(3)

# Mostrar os elementos
q.print_queue()  # Output: [1, 2, 3]

# Retirar o primeiro elemento
element = q.dequeue()
print(element)  # Output: 1

# Mostrar a queue Atual
q.print_queue()  # Output: [2, 3]

# Checar se a queue está vazia ou não
if q.is_empty():
    print("Queue is empty")
else:
    print("Queue is not empty") 