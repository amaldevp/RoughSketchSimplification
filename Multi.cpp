//Code written by Amal Dev Parakkat on 17th Jan 2020 as a reimplementation of our paper "A Delaunay triangulation based approach for cleaning rough sketches" - Amal Dev Parakkat et al.
//You can find the paper in "https://doi.org/10.1016/j.cag.2018.05.011". Do cite it, if you are using the code.
//The implementation is a little bit (very minor) different from the base paper. Please read the changes and assumptions (IMPORTANT).
//The code does not have optimal running time, so it can be used for generating results only. There might be bugs. The code is very simple and straightforward, mostly you yourselves would be able to edit it (send me a copy as well ;-)).

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/Color.h>
#include <CGAL/Triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include<iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#include <vector>
#include <string>
#include <sstream>
using namespace cv;
struct info{
    CGAL::Color c=CGAL::Color(0,0,0,0);
    float val=-10;
    int id=-1;
    int parent=-1;
}inf1;
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_2<K> Vb;
typedef CGAL::Triangulation_face_base_with_info_2<info,K> Fb;
typedef CGAL::Triangulation_data_structure_2<Vb,Fb> Tds;
typedef CGAL::Triangulation_2<K,Tds> Triangulation;
typedef Triangulation::Face_handle Face_handle;
typedef CGAL::Triangulation_2<K,Tds>::Point  point;
typedef CGAL::Delaunay_triangulation_2<K,Tds>  Delaunay;
using namespace cv;
const int alpha_slider_max = 100;
int alpha_slider;
double alpha,beta;
const int alpha_slider_max1 = 100;
int alpha_slider1;
double alpha1,beta1;
char inputname[100];
Delaunay dt;
int minx=9999,miny=9999,maxx=0,maxy=0,wd,sw[100],swi=0,trinin=0,tci,newval=0,newval1=0,newval2=0,newval3=0;
cv::Mat image,oimage,img_skel,filled;
double distance(point a,point b)
{
    float x1,x2,y1,y2;
    x1=a.x();
    x2=b.x();
    y1=a.y();
    y2=b.y();
    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
double seed(Delaunay::Finite_faces_iterator ffi)
{
    if(ffi->info().id==1)
        return -9999;
    return distance(CGAL::circumcenter(ffi->vertex(0)->point(),ffi->vertex(1)->point(),ffi->vertex(2)->point()),ffi->vertex(0)->point());
}
int infinite(Delaunay::Face_handle ffi)
{
    if(ffi->vertex(0)->point()==dt.infinite_vertex()->point())
        return 1;
    if(ffi->vertex(1)->point()==dt.infinite_vertex()->point())
        return 1;
    if(ffi->vertex(2)->point()==dt.infinite_vertex()->point())
        return 1;
    return 0;
}
int startsave=0;
void recurse(Delaunay::Face_handle fh,double len)
{
    if(startsave==1)
        trinin++;
    if(fh->info().id!=1)
    {
        fh->info().id=1;
        if(distance(fh->vertex(0)->point(),fh->vertex(1)->point())>len&&!infinite(fh->neighbor(2)))
            recurse(fh->neighbor(2),len);
        if(distance(fh->vertex(0)->point(),fh->vertex(2)->point())>len&&!infinite(fh->neighbor(1)))
            recurse(fh->neighbor(1),len);
        if(distance(fh->vertex(2)->point(),fh->vertex(1)->point())>len&&!infinite(fh->neighbor(0)))
            recurse(fh->neighbor(0),len);
    }
}

int check_inside(point pt, point *pgn_begin, point *pgn_end, K traits)
{
    switch(CGAL::bounded_side_2(pgn_begin, pgn_end,pt, traits)) {
    case CGAL::ON_BOUNDED_SIDE :
        return 1;
        break;
    case CGAL::ON_BOUNDARY:
        return 1;
        break;
    case CGAL::ON_UNBOUNDED_SIDE:
        return 0;
        break;
    }
}
int fat(point a,point b,point c,point d)
{
    point points[] = { point(a.x(),a.y()),point(b.x(),b.y()),point(c.x(),c.y())};
    return check_inside(point(d.x(),d.y()), points, points+3, K());
}
void thinningIteration(cv::Mat& img, int iter)
{
    CV_Assert(img.channels() == 1);
    CV_Assert(img.depth() != sizeof(uchar));
    CV_Assert(img.rows > 3 && img.cols > 3);
    cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);
    int nRows = img.rows;
    int nCols = img.cols;
    if (img.isContinuous()) {
        nCols *= nRows;
        nRows = 1;
    }
    int x, y;
    uchar *pAbove;
    uchar *pCurr;
    uchar *pBelow;
    uchar *nw, *no, *ne;
    uchar *we, *me, *ea;
    uchar *sw, *so, *se;
    uchar *pDst;
    pAbove = NULL;
    pCurr  = img.ptr<uchar>(0);
    pBelow = img.ptr<uchar>(1);
    for (y = 1; y < img.rows-1; ++y) {
        pAbove = pCurr;
        pCurr  = pBelow;
        pBelow = img.ptr<uchar>(y+1);
        pDst = marker.ptr<uchar>(y);
        no = &(pAbove[0]);
        ne = &(pAbove[1]);
        me = &(pCurr[0]);
        ea = &(pCurr[1]);
        so = &(pBelow[0]);
        se = &(pBelow[1]);
        for (x = 1; x < img.cols-1; ++x) {
            nw = no;
            no = ne;
            ne = &(pAbove[x+1]);
            we = me;
            me = ea;
            ea = &(pCurr[x+1]);
            sw = so;
            so = se;
            se = &(pBelow[x+1]);
            int A  = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
                    (*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
                    (*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
                    (*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
            int B  = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
            int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
            int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);
            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                pDst[x] = 1;
        }
    }
    img &= ~marker;
}
void thinning(const cv::Mat& src, cv::Mat& dst)
{
    dst = src.clone();
    dst /= 255;
    cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
    cv::Mat diff;
    do {
        thinningIteration(dst, 0);
        thinningIteration(dst, 1);
        cv::absdiff(dst, prev, diff);
        dst.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);
    dst *= 255;
}
Mat img1,img2;
std::vector<Point> branches[1000];
std::vector<Vec3b> branchcol;
std::vector<Point> junctions;
std::vector<Point> ends;
int bi=0;
Vec3b strokecol;
int order=0;
std::vector<Point> obranches[1000];
int obi=0;
void recursefill(int i,int j)
{
    if(i<0||j<0||i>img2.rows-1||j>img2.cols-1)
        return;
    Vec3b color;
    img2.at<Vec3b>(Point(j,i)) = strokecol;
    if(order==0)
        branches[bi].push_back(Point(j,i));
    else
        obranches[obi].push_back(Point(j,i));
    int r1;
    color=img2.at<cv::Vec3b>(i-1,j-1);
    r1=color[0];
    if(r1==0)
        recursefill(i-1,j-1);
    color=img2.at<cv::Vec3b>(i,j-1);
    r1=color[0];
    if(r1==0)
        recursefill(i,j-1);
    color=img2.at<cv::Vec3b>(i+1,j-1);
    r1=color[0];
    if(r1==0)
        recursefill(i+1,j-1);
    color=img2.at<cv::Vec3b>(i-1,j);
    r1=color[0];
    if(r1==0)
        recursefill(i-1,j);
    color=img2.at<cv::Vec3b>(i+1,j);
    r1=color[0];
    if(r1==0)
        recursefill(i+1,j);
    color=img2.at<cv::Vec3b>(i-1,j+1);
    r1=color[0];
    if(r1==0)
        recursefill(i-1,j+1);
    color=img2.at<cv::Vec3b>(i,j+1);
    r1=color[0];
    if(r1==0)
        recursefill(i,j+1);
    color=img2.at<cv::Vec3b>(i+1,j+1);
    r1=color[0];
    if(r1==0)
        recursefill(i+1,j+1);
}
int segtype[1000];
Mat prunedskeleton;
void findbranches()
{
    for(int i=0;i<bi;i++)
        branches[i].clear();
    cv::imshow("Skeleton", img_skel);
    imwrite("skel.png",img_skel);
    img1 = cv::imread("skel.png", 1 );
    img2 = cv::imread("skel.png", 1 );
    for(int i=1;i<img1.rows-1;i++)
        for(int j=1;j<img1.cols-1;j++)
        {
            int neigh1=0,neigh2=0,neigh3=0,neigh4=0,neigh5=0,neigh6=0,neigh7=0,neigh8=0;
            cv::Vec3b color=img1.at<cv::Vec3b>(i,j);
            int r1=color.val[0];
            if(r1<250&&color.val[1]<250&&color.val[2]<250)
            {
                color=img1.at<cv::Vec3b>(i-1,j-1);
                r1=color.val[0];
                if(color.val[0]<250&&color.val[1]<250&&color.val[2]<250)
                    neigh1=1;
                color=img1.at<cv::Vec3b>(i,j-1);
                r1=color.val[0];
                if(color.val[0]<250&&color.val[1]<250&&color.val[2]<250)
                    neigh2=1;
                color=img1.at<cv::Vec3b>(i+1,j-1);
                r1=color.val[0];
                if(color.val[0]<250&&color.val[1]<250&&color.val[2]<250)
                    neigh3=1;
                color=img1.at<cv::Vec3b>(i-1,j);
                r1=color.val[0];
                if(color.val[0]<250&&color.val[1]<250&&color.val[2]<250)
                    neigh4=1;
                color=img1.at<cv::Vec3b>(i+1,j);
                r1=color.val[0];
                if(color.val[0]<250&&color.val[1]<250&&color.val[2]<250)
                    neigh5=1;
                color=img1.at<cv::Vec3b>(i-1,j+1);
                r1=color.val[0];
                if(color.val[0]<250&&color.val[1]<250&&color.val[2]<250)
                    neigh6=1;
                color=img1.at<cv::Vec3b>(i,j+1);
                r1=color.val[0];
                if(color.val[0]<250&&color.val[1]<250&&color.val[2]<250)
                    neigh7=1;
                color=img1.at<cv::Vec3b>(i+1,j+1);
                r1=color.val[0];
                if(color.val[0]<250&&color.val[1]<250&&color.val[2]<250)
                    neigh8=1;
                int sum=neigh1+neigh2+neigh3+neigh4+neigh5+neigh6+neigh7+neigh8;
                if(sum==1)
                {
                    ends.push_back(Point(j,i));
                    circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                }
                else
                {
                    if(sum>=3)
                    {
                        if(neigh1+neigh6+neigh5==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh1+neigh6+neigh3==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh1+neigh6+neigh8==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh1+neigh3+neigh6==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh1+neigh3+neigh7==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh1+neigh3+neigh8==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh3+neigh8+neigh1==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh3+neigh8+neigh4==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh3+neigh8+neigh6==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }

                        if(neigh6+neigh8+neigh1==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh6+neigh8+neigh2==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh6+neigh8+neigh3==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh2+neigh4+neigh8==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh6+neigh5+neigh2==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh5+neigh7+neigh1==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                        if(neigh7+neigh4+neigh3==3)
                        {
                            junctions.push_back(Point(j,i));
                            circle(img2,Point(j,i),1,cv::Scalar(255,255,255),CV_FILLED,8,0);
                        }
                    }
                }
            }
        }
    for(int i=1;i<img2.rows-1;i++)
        for(int j=1;j<img2.cols-1;j++)
        {
            cv::Vec3b color=img2.at<cv::Vec3b>(i,j);
            if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
            {
                strokecol[0]=rand()%200+50;
                strokecol[1]=rand()%200+50;
                strokecol[2]=rand()%200+50;
                branchcol.push_back(strokecol);
                branches[bi].push_back(Point(j,i));
                recursefill(i,j);
                bi++;
            }
        }
    std::vector<int> kind[1000];
    for(int i=0;i<bi;i++)
    {
        int entered=0;
        for(int j=0;j<branches[i].size();j++)
        {
            for(int k=0;k<junctions.size();k++)
                if(distance(point(branches[i].at(j).x,branches[i].at(j).y),point(junctions.at(k).x,junctions.at(k).y))<2.3)
                {
                    entered=1;
                    kind[i].push_back(1);
                    circle(img2,branches[i].at(j),1,cv::Scalar(255,0,0),CV_FILLED,8,0);
                }
            for(int k=0;k<ends.size();k++)
                if(distance(point(branches[i].at(j).x,branches[i].at(j).y),point(ends.at(k).x,ends.at(k).y))<2.3)
                {
                    entered=1;
                    kind[i].push_back(2);
                    circle(img2,branches[i].at(j),1,cv::Scalar(0,0,255),CV_FILLED,8,0);
                }
        }
        if(entered==0)
            kind[i].push_back(0);
    }
    Vec3b loop,branch,inter,open;
    loop[0]=0;
    loop[1]=0;
    loop[2]=255;
    branch[0]=0;
    branch[1]=255;
    branch[3]=0;
    inter[0]=255;
    inter[1]=0;
    inter[2]=0;
    open[0]=255;
    open[1]=255;
    open[2]=0;
    for(int i=0;i<bi;i++)
    {
        if(kind[i].at(0)==0)
        {
            segtype[i]=0;
            for(int j=0;j<branches[i].size();j++)
                img2.at<Vec3b>(Point(branches[i].at(j).x,branches[i].at(j).y)) = loop;
        }
        else
        {
            int onef=0,twof=0;
            for(int j=0;j<kind[i].size();j++)
            {
                if(kind[i].at(j)==1)
                    onef=1;
                if(kind[i].at(j)==2)
                    twof=1;
            }
            if(onef==1&&twof==0)
            {
                segtype[i]=2;
                for(int j=0;j<branches[i].size();j++)
                    img2.at<Vec3b>(Point(branches[i].at(j).x,branches[i].at(j).y)) = inter;
            }
            if(onef==1&&twof==1)
            {
                segtype[i]=3;
                for(int j=0;j<branches[i].size();j++)
                    img2.at<Vec3b>(Point(branches[i].at(j).x,branches[i].at(j).y)) = branch;
            }
            if(onef==0&&twof==1)
            {
                segtype[i]=4;
                for(int j=0;j<branches[i].size();j++)
                    img2.at<Vec3b>(Point(branches[i].at(j).x,branches[i].at(j).y)) = open;
            }
        }
    }
}
Mat prunedskel;
void orderpoints()
{
    order=1;
    img2=prunedskel.clone();
    for(int i=0;i<obi;i++)
        obranches[i].clear();
    obi=0;
    std::vector<Point> startpts;
    for(int i=1;i<img2.rows-1;i++)
        for(int j=1;j<img2.cols-1;j++)
        {
            cv::Vec3b color=img2.at<cv::Vec3b>(i,j);
            if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
            {
                int neigh1=0,neigh2=0,neigh3=0,neigh4=0,neigh5=0,neigh6=0,neigh7=0,neigh8=0;
                color=img2.at<cv::Vec3b>(i-1,j-1);
                int r1;
                r1=color.val[0];
                if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
                    neigh1=1;
                color=img2.at<cv::Vec3b>(i,j-1);
                r1=color.val[0];
                if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
                    neigh2=1;
                color=img2.at<cv::Vec3b>(i+1,j-1);
                r1=color.val[0];
                if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
                    neigh3=1;
                color=img2.at<cv::Vec3b>(i-1,j);
                r1=color.val[0];
                if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
                    neigh4=1;
                color=img2.at<cv::Vec3b>(i+1,j);
                r1=color.val[0];
                if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
                    neigh5=1;
                color=img2.at<cv::Vec3b>(i-1,j+1);
                r1=color.val[0];
                if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
                    neigh6=1;
                color=img2.at<cv::Vec3b>(i,j+1);
                r1=color.val[0];
                if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
                    neigh7=1;
                color=img2.at<cv::Vec3b>(i+1,j+1);
                r1=color.val[0];
                if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
                    neigh8=1;
                int sum=neigh1+neigh2+neigh3+neigh4+neigh5+neigh6+neigh7+neigh8;
                if(sum==1)
                {
                    strokecol[0]=rand()%200+50;
                    strokecol[1]=rand()%200+50;
                    strokecol[2]=rand()%200+50;
                    branchcol.push_back(strokecol);
                    obranches[obi].push_back(Point(j,i));
                    startpts.push_back(Point(j,i));
                    recursefill(i,j);
                    obi++;
                }
                else
                    if(sum==2)
                    {
                        if(neigh1+neigh2==2||neigh2+neigh3==2||neigh3+neigh5==2||neigh5+neigh8==2||neigh8+neigh7==2||neigh6+neigh7==2||neigh4+neigh6==2||neigh1+neigh4==2)
                        {
                            strokecol[0]=rand()%200+50;
                            strokecol[1]=rand()%200+50;
                            strokecol[2]=rand()%200+50;
                            branchcol.push_back(strokecol);
                            obranches[obi].push_back(Point(j,i));
                            startpts.push_back(Point(j,i));
                            recursefill(i,j);
                            obi++;
                        }
                    }
            }
        }
    int cbi=obi-1;
    std::vector<int> closedindex;
    for(int i=1;i<img2.rows-1;i++)
        for(int j=1;j<img2.cols-1;j++)
        {
            cv::Vec3b color=img2.at<cv::Vec3b>(i,j);
            if(color.val[0]==0&&color.val[1]==0&&color.val[2]==0)
            {
                strokecol[0]=rand()%200+50;
                strokecol[1]=rand()%200+50;
                strokecol[2]=rand()%200+50;
                branchcol.push_back(strokecol);
                startpts.push_back(Point(j,i));
                obranches[obi].push_back(Point(j,i));
                recursefill(i,j);
                closedindex.push_back(obi);
                obi++;
            }
        }
    order=0;
    std::vector<int> kind[1000];
    Mat temp=img2.clone();
    std::vector<Point> skip;
    for(int i=0;i<startpts.size();i++)
    {
        skip.clear();
        skip.push_back(startpts.at(i));
        int startx=startpts.at(i).x;
        int starty=startpts.at(i).y;
        Vec3b ocol=temp.at<cv::Vec3b>(startx,starty);
        Vec3b whitecol;
        whitecol[0]=255;
        whitecol[1]=255;
        whitecol[2]=255;
        int co=0;
        while(1)
        {
            whitecol[0]=255;
            whitecol[1]=255;
            whitecol[2]=255;
            co++;
            int n1=0,n2=0,n3=0,n4=0,n5=0,n6=0,n7=0,n8=0;
            Vec3b color;
            color=temp.at<cv::Vec3b>(starty-1,startx-1);
            if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                n1=1;
            color=temp.at<cv::Vec3b>(starty-1,startx);
            if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                n2=1;
            color=temp.at<cv::Vec3b>(starty,startx-1);
            if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                n3=1;
            color=temp.at<cv::Vec3b>(starty+1,startx-1);
            if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                n4=1;
            color=temp.at<cv::Vec3b>(starty+1,startx);
            if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                n5=1;
            color=temp.at<cv::Vec3b>(starty+1,startx+1);
            if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                n6=1;
            color=temp.at<cv::Vec3b>(starty-1,startx+1);
            if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                n7=1;
            color=temp.at<cv::Vec3b>(starty,startx+1);
            if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                n8=1;
            temp.at<Vec3b>(Point(startx-1,starty-1)) = whitecol;
            temp.at<Vec3b>(Point(startx-1,starty)) = whitecol;
            temp.at<Vec3b>(Point(startx-1,starty+1)) = whitecol;
            temp.at<Vec3b>(Point(startx,starty-1)) = whitecol;
            temp.at<Vec3b>(Point(startx,starty)) = whitecol;
            temp.at<Vec3b>(Point(startx,starty+1)) = whitecol;
            temp.at<Vec3b>(Point(startx+1,starty-1)) = whitecol;
            temp.at<Vec3b>(Point(startx+1,starty)) = whitecol;
            temp.at<Vec3b>(Point(startx+1,starty+1)) = whitecol;
            int min=99999;
            Point nextp;
            for(int j=0;j<obranches[i].size();j++)
            {
                color=temp.at<cv::Vec3b>(obranches[i].at(j).y,obranches[i].at(j).x);
                if(color.val[0]!=255||color.val[1]!=255||color.val[2]!=255)
                {
                    double dist=distance(point(startx,starty),point(obranches[i].at(j).x,obranches[i].at(j).y));
                    if(dist<min)
                    {
                        min=dist;
                        nextp=obranches[i].at(j);
                    }
                }
            }
            if(min==99999)
            {
                obranches[i].clear();
                for(int k=0;k<skip.size();k++)
                    obranches[i].push_back(skip.at(k));
                if(n1+n2+n3+n4+n5+n6+n7+n8!=0)
                {
                    if(n1==1)
                        obranches[i].push_back(Point(startx-1,starty-1));
                    if(n1==2)
                        obranches[i].push_back(Point(startx,starty-1));
                    if(n1==3)
                        obranches[i].push_back(Point(startx-1,starty));
                    if(n1==4)
                        obranches[i].push_back(Point(startx-1,starty+1));
                    if(n1==5)
                        obranches[i].push_back(Point(startx,starty+1));
                    if(n1==6)
                        obranches[i].push_back(Point(startx+1,starty+1));
                    if(n1==7)
                        obranches[i].push_back(Point(startx+1,starty-1));
                    if(n1==8)
                        obranches[i].push_back(Point(startx+1,starty));
                }
                break;
            }
            else
            {
                skip.push_back(nextp);
                startx=nextp.x;
                starty=nextp.y;
            }
        }
    }
    for(int i=0;i<obi;i++)
    {
        int entered=0;
        if(obranches[i].size()==1)
        {
            Point onep,twop;
            int fl=0;
            for(int k=0;k<junctions.size();k++)
                if(distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(junctions.at(k).x,junctions.at(k).y))<2.85)
                {
                    onep=junctions.at(k);
                    fl=1;
                }
            if(fl==1)
                for(int k=0;k<junctions.size();k++)
                    if(distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(junctions.at(k).x,junctions.at(k).y))<2.85&&
                            distance(point(onep.x,onep.y),point(junctions.at(k).x,junctions.at(k).y))>0.01)
                        twop=junctions.at(k);
                    else
                    {
                        if(fl==0)
                        {
                            for(int k=0;k<ends.size();k++)
                                if(distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(ends.at(k).x,ends.at(k).y))<2.85)
                                {
                                    onep=ends.at(k);
                                    fl=1;
                                }
                        }
                        else
                        {
                            for(int k=0;k<ends.size();k++)
                                if(distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(ends.at(k).x,ends.at(k).y))<2.85&&
                                        distance(point(onep.x,onep.y),point(ends.at(k).x,ends.at(k).y))>0.01)
                                    twop=ends.at(k);
                        }
                    }
            obranches[i].insert(obranches[i].begin(),onep);
            obranches[i].insert(obranches[i].end(),twop);
        }
        else
        {
            Point onep,twop;
            // std::cout<<obranches[i].at(0)<<" "<<obranches[i].at(obranches[i].size()-1)<<"\n";
            int min1=9999,min2=9999;
            if(i<=cbi)
            {
                for(int k=0;k<junctions.size();k++)
                    if(distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(junctions.at(k).x,junctions.at(k).y))<min1)
                    {
                        min1=distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(junctions.at(k).x,junctions.at(k).y));
                        onep=junctions.at(k);
                    }
                for(int k=0;k<ends.size();k++)
                    if(distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(ends.at(k).x,ends.at(k).y))<min1)
                    {
                        min1=distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(ends.at(k).x,ends.at(k).y));
                        onep=ends.at(k);
                    }
                for(int k=0;k<junctions.size();k++)
                    if(distance(point(obranches[i].at(obranches[i].size()-1).x,obranches[i].at(obranches[i].size()-1).y),point(junctions.at(k).x,junctions.at(k).y))<min2)
                    {
                        min2=distance(point(obranches[i].at(obranches[i].size()-1).x,obranches[i].at(obranches[i].size()-1).y),point(junctions.at(k).x,junctions.at(k).y));
                        twop=junctions.at(k);
                    }
                for(int k=0;k<ends.size();k++)
                    if(distance(point(obranches[i].at(obranches[i].size()-1).x,obranches[i].at(obranches[i].size()-1).y),point(ends.at(k).x,ends.at(k).y))<min2)
                    {
                        min2=distance(point(obranches[i].at(obranches[i].size()-1).x,obranches[i].at(obranches[i].size()-1).y),point(ends.at(k).x,ends.at(k).y));
                        twop=ends.at(k);
                    }
                if(distance(point(obranches[i].at(0).x,obranches[i].at(0).y),point(onep.x,onep.y))<4.1)
                    obranches[i].insert(obranches[i].begin(),onep);
                if(distance(point(obranches[i].at(obranches[i].size()-1).x,obranches[i].at(obranches[i].size()-1).y),point(twop.x,twop.y))<4.1)
                    obranches[i].insert(obranches[i].end(),twop);
            }
        }
    }
    for(int i=0;i<closedindex.size();i++)
        obranches[closedindex.at(i)].insert(obranches[closedindex.at(i)].end(),obranches[closedindex.at(i)].at(0));
}
int alpha3, alpha_slider3,alpha_slider_max3,beta3;
std::vector<Point> resultpts[1000];
int rpi=0;
void smoothshape()
{
    Mat smoothlines(img1.rows,img1.cols, CV_8UC3, Scalar(255,255,255));
    std::vector<Point> smooth[1000];
    int si=0;
    Vec3b brush;
    brush[0]=0;
    brush[1]=0;
    brush[2]=255;

    for(int i=0;i<obi;i++)
    {
        smooth[si].push_back(obranches[i].at(0));
        if(obi<4)
            smooth[si].push_back(obranches[i].at(0));
        else
        {
            float ox=obranches[i].at(0).x;
            float oy=obranches[i].at(0).y;
            float x=obranches[i].at(0).x;
            float y=obranches[i].at(0).y;
            int nind=1;
            int fl=0;
            while(1)
            {
                float dx=obranches[i].at(nind).x-ox;
                float dy=obranches[i].at(nind).y-oy;
                int steps;
                x=ox;
                y=oy;
                if(abs(dx)>abs(dy))
                    steps=abs(dx);
                else
                    steps=abs(dy);
                float xinc=dx/(float)steps;
                float yinc=dy/(float)steps;
                for(int v=0;v<steps;v++)
                {
                    x=x+xinc;
                    y=y+yinc;
                    brush=filled.at<Vec3b>(Point(x,y));
                    if(brush[0]>50&&brush[1]>50&&brush[2]>50)
                        fl=1;
                }
                if(fl==1||distance(point((int)ox,(int)oy),point(obranches[i].at(nind).x,obranches[i].at(nind).y))>100-alpha_slider3)
                {
                    smooth[si].push_back(obranches[i].at(nind-1));
                    ox=obranches[i].at(nind-1).x;
                    oy=obranches[i].at(nind-1).y;
                    fl=0;
                }
                if(nind==obranches[i].size()-1)
                    break;
                nind++;
            }
        }
        smooth[si].push_back(obranches[i].at(obranches[i].size()-1));
        smooth[si].push_back(obranches[i].at(obranches[i].size()-1));
        si++;
    }
    for(int i=0;i<si;i++)
    {
        resultpts[i].clear();
        int fl=1;
        for(int j=0;j<smooth[i].size();j++)
        {
            resultpts[i].push_back(smooth[i].at(j));
            if(j<smooth[i].size()-1)
                if(smooth[i].at(j+1).x>0&&smooth[i].at(j+1).y>0&&smooth[i].at(j+1).x<image.cols&&smooth[i].at(j+1).y<image.rows)
                    if(smooth[i].at(j).x>0&&smooth[i].at(j).y>0&&smooth[i].at(j).x<image.cols&&smooth[i].at(j).y<image.rows)
                        line(smoothlines,smooth[i].at(j),smooth[i].at(j+1),Scalar(0,0,255),1,8);
        }
    }
    rpi=si;
    cv::imshow("Smooth", smoothlines);
}

static void on_trackbar3( int, void* )
{
    alpha3 = (double) alpha_slider3/alpha_slider_max3 ;
    beta3 = ( 1.0 - alpha3 );
    if(alpha_slider3>0)
        smoothshape();
}
int ft=0;
void onmousesmooth(int event, int x, int y, int flags, void* param)
{
    if(event==2)
    {
        std::string s=inputname;
        if(ft==0)
        {
            s=s+".html";
            ft=1;
        }
        strcpy(inputname, s.c_str());
        FILE *fp=fopen(inputname,"w");
        fprintf(fp,"<div class=\"contain-demo\">\n<svg\nwidth=\"210mm\"\nheight=\"297mm\"\nviewBox=\"0 0 1000 1000\"\nversion=\"1.1\"\nid=\"svg8\"\ninkscape:version=\"0.92.3 (2405546, 2018-03-11)\""
                   "\nsodipodi:docname=\"drawing.svg\">\n<g\ninkscape:label=\"Layer 1\"\ninkscape:groupmode=\"layer\"\nid=\"layer1\">\n");
        for(int i=0;i<rpi;i++)
        {
            int fl=1;
            fprintf(fp,"<path\nstyle=\"fill:none;stroke:#000000;stroke-width:1px;stroke-linecap:butt;stroke-linejoin:miter;stroke-opacity:1\"\nd=\"M %d,%d C ",resultpts[i].at(0).x,resultpts[i].at(0).y);
            for(int j=0;j<resultpts[i].size();j++)
            {
                if(resultpts[i].at(j).x>0&&resultpts[i].at(j).y>0&&resultpts[i].at(j).x<image.cols&&resultpts[i].at(j).y<image.rows)
                {
                    if(fl==0)
                        fprintf(fp," %d,%d",resultpts[i].at(j).x,resultpts[i].at(j).y);
                    else
                    {
                        fprintf(fp,"%d,%d",resultpts[i].at(j).x,resultpts[i].at(j).y);
                        fl=0;
                    }
                }
            }
            if(resultpts[i].at(resultpts[i].size()-1).x>0&&resultpts[i].at(resultpts[i].size()-1).y>0&&resultpts[i].at(resultpts[i].size()-1).x<image.cols&&resultpts[i].at(resultpts[i].size()-1).y<image.rows)
                fprintf(fp," %d,%d",resultpts[i].at(resultpts[i].size()-1).x,resultpts[i].at(resultpts[i].size()-1).y);
            else
            {     if(resultpts[i].size()>1)
                    if(resultpts[i].at(resultpts[i].size()-2).x>0&&resultpts[i].at(resultpts[i].size()-2).y>0&&resultpts[i].at(resultpts[i].size()-2).x<image.cols&&resultpts[i].at(resultpts[i].size()-2).y<image.rows)

                        fprintf(fp," %d,%d",resultpts[i].at(resultpts[i].size()-2).x,resultpts[i].at(resultpts[i].size()-2).y);
                    else
                        fprintf(fp," %d,%d",resultpts[i].at(0).x,resultpts[i].at(0).y);
            }
            fprintf(fp,"\"\nid=\"path815\"\ninkscape:connector-curvature=\"0\"\nsodipodi:nodetypes=\"csc\" />\n");
        }
        fprintf(fp,"</g>\n</svg>\n</div>\n<p class=\"p\">Clean Sketch.</p>");
        fclose(fp);
    }
}

void onmouseskel(int event, int x, int y, int flags, void* param)
{
    if(event==2)
    {
        orderpoints();
        cv::namedWindow("Smooth", cv::WINDOW_AUTOSIZE );
        createTrackbar("Smooth", "Smooth", &alpha_slider3, 100, on_trackbar3);
        cv::imshow("Smooth", prunedskeleton);
        setMouseCallback("Smooth", onmousesmooth, &image);
        smoothshape();
    }
}
int alpha2, alpha_slider2,alpha_slider_max2,beta2;

void showpruned()
{
    Vec3b brush;
    brush[0]=0;
    brush[1]=0;
    brush[2]=0;
    Mat pruned(img1.rows,img1.cols, CV_8UC3, Scalar(255,255,255));
    for(int i=0;i<bi;i++)
        if(segtype[i]!=3||branches[i].size()>=alpha_slider2)
            for(int j=0;j<branches[i].size();j++)
                pruned.at<Vec3b>(Point(branches[i].at(j).x,branches[i].at(j).y)) =brush;
    prunedskel=pruned.clone();
    prunedskeleton=pruned.clone();
    cv::imshow("Skeleton", pruned);
    imwrite("pruned.png",pruned);
}

static void on_trackbar2( int, void* )
{
    alpha2 = (double) alpha_slider2/alpha_slider_max2 ;
    beta2 = ( 1.0 - alpha2 );
    if(alpha_slider2>0)
        showpruned();
}

void onmouse(int event, int x, int y, int flags, void* param)
{
    Mat &img = *((Mat*)(param));
    if(event==2)
    {
        filled=image.clone();
        imwrite("Filled.png",filled);
        Mat im_gray;
        Mat blank(img1.rows,img1.cols, CV_8UC3, Scalar(255,255,255));
        img_skel=blank.clone();
        cvtColor(image,im_gray,CV_RGB2GRAY);
        Mat img_bw = im_gray > 128;
        cv::subtract(cv::Scalar::all(255),img_bw,img_bw);
        thinning(img_bw,img_skel);
        Mat sub_mat = Mat::ones(img_skel.size(), img_skel.type())*255;
        subtract(sub_mat,img_skel,img_skel);
        bitwise_and(img_bw,img_skel,img_bw);
        cv::namedWindow("Skeleton", cv::WINDOW_AUTOSIZE );
        createTrackbar("Skeleton Pruning", "Skeleton", &alpha_slider2, 200, on_trackbar2);
        setMouseCallback("Skeleton", onmouseskel, &image);
        findbranches();
    }
}
void findfilledshape(double par,double len)
{
    trinin=0;newval=0;newval1=0;newval2=0;newval3=0;
    image=oimage.clone();
    startsave=0;
    Delaunay::Finite_faces_iterator ffi=dt.finite_faces_begin();
    ffi=dt.finite_faces_begin();
    do{
        ffi->info().id=-1;
        Delaunay::Face_handle(ffi)->info().id=-1;
    }while(++ffi!=dt.finite_faces_end());
    ffi=dt.finite_faces_begin();
    do{
        if((infinite(ffi->neighbor(0))&&distance(ffi->vertex(2)->point(),ffi->vertex(1)->point())>len)||(infinite(ffi->neighbor(1))&&distance(ffi->vertex(2)->point(),ffi->vertex(0)->point())>len)||(infinite(ffi->neighbor(2))&&distance(ffi->vertex(0)->point(),ffi->vertex(1)->point())>len))
            recurse(Delaunay::Face_handle(ffi),len);
    }while(++ffi!=dt.finite_faces_end());
    ffi=dt.finite_faces_begin();
    Delaunay::Finite_faces_iterator fmax;
    for(int i=0;i<par;i++)
    {
        int e=0;
        double max=-9999;
        ffi=dt.finite_faces_begin();
        do{
            if(seed(ffi)>max&&fat(ffi->vertex(2)->point(),ffi->vertex(1)->point(),ffi->vertex(0)->point(),CGAL::circumcenter(ffi->vertex(2)->point(),ffi->vertex(1)->point(),ffi->vertex(0)->point())))
            {
                e=1;
                fmax=ffi;
                max=distance(CGAL::circumcenter(ffi->vertex(2)->point(),ffi->vertex(1)->point(),ffi->vertex(0)->point()),ffi->vertex(0)->point());
            }
        }while(++ffi!=dt.finite_faces_end());
        if(e!=1)
            break;
        startsave=1;
        newval=trinin;
        recurse(Delaunay::Face_handle(fmax),len);
    }
    ffi=dt.finite_faces_begin();
    do{
        point a,b,c;
        int e=0;
        a=ffi->vertex(0)->point();
        b=ffi->vertex(1)->point();
        c=ffi->vertex(2)->point();
        if(ffi->info().id!=1)
        {

            std::vector<Point> tmp;
            tmp.push_back(Point(a.x(),a.y()));
            tmp.push_back(Point(b.x(),b.y()));
            tmp.push_back(Point(c.x(),c.y()));
            const Point* elementPoints[1] = { &tmp[0] };
            int numberOfPoints = (int)tmp.size();
            fillPoly (image, elementPoints, &numberOfPoints, 1, Scalar (0, 0, 0), 8);
        }
    }while(++ffi!=dt.finite_faces_end());
    cv::imshow("Filled Shape", image);
}
int length=0,mask=0;
static void on_trackbar( int, void* )
{
    alpha = (double) alpha_slider/alpha_slider_max ;
    beta = ( 1.0 - alpha );
    if(alpha_slider>-1)
    {
        findfilledshape(mask,alpha_slider);
        length=alpha_slider;
    }
}
static void on_trackbar1( int, void* )
{
    alpha1 = (double) alpha_slider1/alpha_slider_max1 ;
    beta1 = ( 1.0 - alpha1 );
    if(alpha_slider1>0)
    {
        findfilledshape(alpha_slider1,length);
        mask=alpha_slider1;
    }
}
int main(int argc, char **argv)
{
    float sp;
    image = cv::imread( argv[1], 1 );
    oimage = cv::imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    copyMakeBorder(image,image, 2,2,2,2, BORDER_CONSTANT, Scalar(255,255,255) );
    copyMakeBorder(oimage,oimage, 2,2,2,2, BORDER_CONSTANT, Scalar(255,255,255));
    strcpy(inputname,argv[1]);
    int sizex,sizey;
    if(image.rows>image.cols)
    {
        sizey=700;
        sizex=(int)(image.cols*(700.0/image.rows));
    }
    else
    {
        sizex=700;
        sizey=(int)(image.rows*(700.0/image.cols));
    }
    Size size(sizex,sizey);
    resize(image,image,size);
    resize(oimage,oimage,size);
    cv::namedWindow("Filled Shape", cv::WINDOW_AUTOSIZE );
    setMouseCallback("Filled Shape", onmouse, &image);
    cv::imshow("Filled Shape", image);
    for(int i=0; i<image.rows; i++)
        for(int j=0; j<image.cols; j++)
            if(image.at<cv::Vec3b>(i,j)[0]<200)
                dt.insert(point(j,i));
    char TrackbarName[50];
    sprintf( TrackbarName, "Alpha x %d", 10 );
    createTrackbar("Growing Parameter", "Filled Shape", &alpha_slider, alpha_slider_max, on_trackbar );
    on_trackbar( alpha_slider, 0 );
    createTrackbar("Masking Regions", "Filled Shape", &alpha_slider1, 200, on_trackbar1);
    cv::imshow("Filled Shape", image);
    cv::waitKey(0);
    return 0;
}
