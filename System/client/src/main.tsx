import React from 'react';
import ReactDOM from 'react-dom/client';
import { Theme } from '@radix-ui/themes';
import { Toaster } from "react-hot-toast";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from 'react-router-dom';
import Router from './router';
import './index.css';
import '@radix-ui/themes/styles.css';

const queryClient = new QueryClient();

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <Theme appearance="dark" accentColor="iris">
        <RouterProvider router={Router} />
      </Theme>
      <Toaster toastOptions={{ className: "dark:bg-zinc-950 dark:text-white" }} />
    </QueryClientProvider>
  </React.StrictMode>,
)
