#ifndef PATTERNINFO_H
#define PATTERNINFO_H

#include <QMap>
#include <QVector>
#include <QSet>
#include <QString>
#include <QTime>
#include <QImage>
#include <QRect>
#include <QtGui>

class PatternInfo
{
 public:
    typedef QMap<QString,QPoint> PersonsType;
    typedef QVector<std::pair<QPoint,QImage> > ImagesType;
    PersonsType persons;
    PersonsType unrecPersons;
    ImagesType unrecImages;
    QTime creationTime;

    PatternInfo(){
    }

    PatternInfo(const PersonsType& personsMap,
                const PersonsType& unrecPersonsMap,
                const ImagesType& unrecImagesArg):
            persons(personsMap),
            unrecPersons(unrecPersonsMap),
            unrecImages(unrecImagesArg),
            creationTime(QTime::currentTime())
    {
        for(PersonsType::const_iterator iter=persons.begin();iter!=persons.end();++iter){
            PersonsType::const_iterator duplcatedPerson=unrecPersons.find(iter.key());
            if(duplcatedPerson!=unrecPersons.end()){
                unrecImages.push_back(std::make_pair(unrecPersons[iter.key()],QImage()));
                unrecPersons.remove(iter.key());
            }
        }
    }

    bool operator==(const PatternInfo& rhs){
        QSet<QString> totalFoundPersons,rhsTotalFoundPersons;
        for(PersonsType::const_iterator iter=persons.begin();iter!=persons.end();++iter){
            totalFoundPersons.insert(iter.key());
        }
        for(PersonsType::const_iterator iter=unrecPersons.begin();iter!=unrecPersons.end();++iter){
            totalFoundPersons.insert(iter.key());
        }
        for(PersonsType::const_iterator iter=rhs.persons.begin();iter!=rhs.persons.end();++iter){
            rhsTotalFoundPersons.insert(iter.key());
        }
        for(PersonsType::const_iterator iter=rhs.unrecPersons.begin();iter!=rhs.unrecPersons.end();++iter){
            rhsTotalFoundPersons.insert(iter.key());
        }
        return totalFoundPersons==rhsTotalFoundPersons;
        /*
        return (persons==rhs.persons) &&
                ((unrecpersons==rhs.unrecpersons);
        //
                */
    }
    bool operator!=(const PatternInfo& rhs){
        return !(*this==rhs);
    }
    int get_total_persons_count(){
        return persons.size()+unrecPersons.size()+unrecImages.size();
    }
};


#endif // PATTERNINFO_H
