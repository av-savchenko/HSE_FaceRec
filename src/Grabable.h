#ifndef GRABABLE_H
#define GRABABLE_H

class QOpenCVWidget;

class Grabable
{
public:
    Grabable();
    virtual void grab(QOpenCVWidget*)=0;
    virtual ~Grabable(){}

    template <class T> static Grabable* Create(){
        return new T();
    }
};

#endif // GRABABLE_H
