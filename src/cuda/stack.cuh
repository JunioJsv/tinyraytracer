#ifndef STACK_CUH
#define STACK_CUH

#include "cuda_defs.cuh"

template<typename T, int Capacity>
class Stack
{
public:
    Stack() = default;

    GPU bool Push(
        T &&item
    )
    {
        if (IsFull()) return false;

        items[++top] = item;
        return true;
    }

    GPU bool Pop(
        T &out
    )
    {
        if (IsEmpty()) return false;

        out = items[top--];

        return true;
    }

    GPU bool IsEmpty() const
    {
        return top < 0;
    }

    GPU bool IsFull() const
    {
        return top >= Capacity - 1;
    }

    GPU unsigned int Size() const
    {
        if (top < 0) return 0;

        return top + 1;
    }

private:
    int top{-1};
    T items[Capacity]{};
};

#endif
