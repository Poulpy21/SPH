
#include "simplyLinkedList.hpp"
#include "boid.hpp"
#include "boidGroup.hpp"


template <typename T>
__host__ __device__ Node<T>::Node(T *data, Node<T> *next) :
    data(data), next(next) 
{}

template <typename T>
__host__ __device__ Node<T>::Node(const Node<T>& node) :
    data(node.data), next(node.next) 
{}

template <typename T>
__host__ __device__ Node<T>& Node<T>::operator= (const Node<T>& node)
{
    this->data = node.data; 
    this->next = node.next; 
    return *this;
}

template <typename T>
__host__ __device__ Node<T>::~Node() {}

template <typename T>
__host__ __device__ List<T>::List() : first(0), last(0), readMutex(), writeMutex(), nReaders(0) {}

template <typename T>
__host__ __device__ List<T>::List(const List& list) : first(list.first), last(list.last), readMutex(list.readMutex), writeMutex(list.writeMutex), nReaders(list.nReaders) {}

template <typename T>
__host__ __device__ List<T>::~List() {
    this->readMutex.unlock();
    this->writeMutex.unlock();
}

template <typename T>
__host__ __device__ void List<T>::push_front(T *data) {
    
    writeMutex.lock();
    {
        if(first == 0 && last == 0) {
            Node<T> *bn = new Node<T>(data, 0);
            first = bn;
            last = bn;
        }
        else {
            Node<T> *bn = new Node<T>(data, first);
            first = bn;
        }
    }
    writeMutex.unlock();
}

template <typename T>
__host__ __device__ T* List<T>::pop_front() {
    
    if(first == 0)
        return 0;
    
    T *dataBuffer;

    writeMutex.lock();
    {
        dataBuffer = first->data;
        Node<T> *nextNode = first->next;

        delete first;

        last = (first == last ? 0 : last);
        first = nextNode;
    }
    writeMutex.unlock();

    return dataBuffer;
}

template <typename T>
__host__ __device__ void List<T>::push_back(T *data) {

    writeMutex.lock();
    {
        if(first == 0 && last == 0) {
            Node<T> *bn = new Node<T>(data, 0);
            first = bn;
            last = bn;
        }
        else {
            Node<T> *bn = new Node<T>(data, 0);
            last->next = bn;
            last = bn;
        }
    } 
    writeMutex.unlock();
}

template class List<char>;
template class List<Boid>;
template class List<BoidGroup>;

