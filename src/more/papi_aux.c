#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "papi.h"

#define NUM_EVENTS 3

long long values[NUM_EVENTS], min_values[NUM_EVENTS];
int Events[NUM_EVENTS] = { PAPI_L1_DCM, PAPI_L2_DCM, PAPI_L3_TCM};
int num_hwcntrs;

int papi_init (){

  PAPI_library_init (PAPI_VER_CURRENT);
  /* Get the number of hardware counters available */
  if ((num_hwcntrs = PAPI_num_counters()) <= PAPI_OK)  {
    printf ("PAPI error getting number of available hardware counters!\n");
    return 0;
  }    
  // We will be using at most NUM_EVENTS counters
  if (num_hwcntrs >= NUM_EVENTS) {
    num_hwcntrs = NUM_EVENTS;
  } else {
    printf ("Error: there aren't enough counters to monitor %d events!\n", NUM_EVENTS);
    return 0;
  }
  return 1;
}

int papi_start() {
  /* Start counting events */
  if (PAPI_start_counters(Events, num_hwcntrs) != PAPI_OK) {
    printf ("PAPI error starting counters %d %d!\n",num_hwcntrs, NUM_EVENTS);
    return 0;
  }
  return 1;
}


void print_papi(int size){
    /* Stop counting events */
    if (PAPI_stop_counters(values, NUM_EVENTS) != PAPI_OK) {
      printf ("PAPI error stoping counters!\n");
    }
    //Size matrix C
    int total_elements = size*size;

    for (int i=0 ; i< NUM_EVENTS ; i++) 
      min_values[i] = values[i];


    // output PAPI counters' values
    for (int i=0 ; i< NUM_EVENTS ; i++) {
      char EventCodeStr[PAPI_MAX_STR_LEN];

      if (PAPI_event_code_to_name(Events[i], EventCodeStr) == PAPI_OK) {
        printf (";%lld", min_values[i]);
      } else {
        printf ("PAPI UNKNOWN EVENT = %lld\n", min_values[i]);
      }
    }
    printf("\n");

#if NUM_EVENTS >1
    // evaluate CPI and Texec here
      if ((Events[0] == PAPI_TOT_CYC) && (Events[1] == PAPI_TOT_INS)) {
        float CPI = ((float) min_values[0]) / ((float) min_values[1]);
        float CPE = ((float) min_values[0]) / ((float) total_elements);
        long long Texec = (long long) (((float)min_values[0])/2e3);
        //printf ("%lld\n",Texec);
      }
#endif
}