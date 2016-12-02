#ifndef GRABABLEFACTORY_H
#define GRABABLEFACTORY_H

#include "Grabable.h"

#include <QMap>
#include <QString>

class GrabableFactory
{
public:
    typedef Grabable* (*GrabableCreator)();

    static GrabableFactory& Instance();
    Grabable* create(const QString& name);

    void registerCreator(const QString& name, GrabableCreator grabableCreator){
        nameToCreatorMap.insert(name,grabableCreator);
    }
    QList<QString> getNames(){
        return nameToCreatorMap.keys();
    }
private:
    GrabableFactory(){}
    GrabableFactory(const GrabableFactory&);
    GrabableFactory& operator=(const GrabableFactory&);

    QMap<QString,GrabableCreator> nameToCreatorMap;
};

#endif // GRABABLEFACTORY_H
