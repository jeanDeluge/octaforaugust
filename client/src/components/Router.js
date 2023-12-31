import React from "react";
import { Route, Routes } from 'react-router-dom'
import { ROUTE_PATH_LIST } from '../Constant'

function RouteList(){
    return(
        <>
        <Routes>
            <Route path={ROUTE_PATH_LIST.Main.path} element={<ROUTE_PATH_LIST.Main.component/>}/>
            
            <Route path={ROUTE_PATH_LIST.LibrarianMain.path} element={<ROUTE_PATH_LIST.LibrarianMain.component/>}>
                <Route path={ROUTE_PATH_LIST.LibrarianLiveOrganization.path} element={<ROUTE_PATH_LIST.LibrarianLiveOrganization.component/>}/>
                <Route path={ROUTE_PATH_LIST.LibrarianCheckCollection.path} element={<ROUTE_PATH_LIST.LibrarianCheckCollection.component/>}/>
                <Route path={ROUTE_PATH_LIST.LibrarianOrganization.path} element={<ROUTE_PATH_LIST.LibrarianOrganization.component/>}>
                    <Route path={ROUTE_PATH_LIST.OrganizeAll.path} element={<ROUTE_PATH_LIST.OrganizeAll.component/>} />
                    <Route path={ROUTE_PATH_LIST.OrganizeOne.path} element={<ROUTE_PATH_LIST.OrganizeOne.component/>} />
                    <Route path={ROUTE_PATH_LIST.OrganizeNow.path} element={<ROUTE_PATH_LIST.OrganizeNow.component/>} />
                </Route>
            </Route>
            <Route path={ROUTE_PATH_LIST.LibraryUserMain.path} element={<ROUTE_PATH_LIST.LibraryUserMain.component/>}>
                <Route path={ROUTE_PATH_LIST.FindBook.path} element={<ROUTE_PATH_LIST.FindBook.component/>}/>
                <Route path={ROUTE_PATH_LIST.RecommendBook.path} element={<ROUTE_PATH_LIST.RecommendBook.component/>}/>
            </Route>
        </Routes>
        </>
    );
}


export default RouteList;