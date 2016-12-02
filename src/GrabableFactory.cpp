#include "GrabableFactory.h"

GrabableFactory& GrabableFactory::Instance(){
    static GrabableFactory grabableFactory;
    return grabableFactory;
}
Grabable* GrabableFactory::create(const QString& name){
    GrabableCreator grabableCreator=nameToCreatorMap[name];
    if(grabableCreator)
        return grabableCreator();
    else
        return 0;
}
