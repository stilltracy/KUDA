
#ifdef __cplusplus
#include <cstddef>
namespace std {
#else
#include <stddef.h>
#include <stdbool.h>
#endif


#define CPP0X( feature )


typedef enum memory_order {
    memory_order_relaxed, memory_order_acquire, memory_order_release,
    memory_order_acq_rel, memory_order_seq_cst
} memory_order;


typedef struct atomic_flag
{
#ifdef __cplusplus
    bool test_and_set( memory_order = memory_order_seq_cst ) volatile;
    void clear( memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;

    CPP0X( atomic_flag() = default; )
    CPP0X( atomic_flag( const atomic_flag& ) = delete; )
    atomic_flag& operator =( const atomic_flag& ) CPP0X(=delete);

CPP0X(private:)
#endif
    bool __f__;
} atomic_flag;

#define ATOMIC_FLAG_INIT { false }

#ifdef __cplusplus
extern "C" {
#endif

extern bool atomic_flag_test_and_set( volatile atomic_flag* );
extern bool atomic_flag_test_and_set_explicit
( volatile atomic_flag*, memory_order );
extern void atomic_flag_clear( volatile atomic_flag* );
extern void atomic_flag_clear_explicit
( volatile atomic_flag*, memory_order );
extern void atomic_flag_fence
( const volatile atomic_flag*, memory_order );
extern void __atomic_flag_wait__
( volatile atomic_flag* );
extern void __atomic_flag_wait_explicit__
( volatile atomic_flag*, memory_order );
extern volatile atomic_flag* __atomic_flag_for_address__
( const volatile void* __z__ )
__attribute__((const));

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

inline bool atomic_flag::test_and_set( memory_order __x__ ) volatile
{ return atomic_flag_test_and_set_explicit( this, __x__ ); }

inline void atomic_flag::clear( memory_order __x__ ) volatile
{ atomic_flag_clear_explicit( this, __x__ ); }

inline void atomic_flag::fence( memory_order __x__ ) const volatile
{ atomic_flag_fence( this, __x__ ); }

#endif


#define _ATOMIC_LOAD_( __a__, __x__ ) \
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \
   __atomic_flag_wait_explicit__( __g__, __x__ ); \
   __typeof__((__a__)->__f__) __r__ = *__p__; \
   atomic_flag_clear_explicit( __g__, __x__ ); \
   __r__; })

#define _ATOMIC_STORE_( __a__, __m__, __x__ ) \
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \
   __typeof__(__m__) __v__ = (__m__); \
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \
   __atomic_flag_wait_explicit__( __g__, __x__ ); \
   *__p__ = __v__; \
   atomic_flag_clear_explicit( __g__, __x__ ); \
   __v__; })

#define _ATOMIC_MODIFY_( __a__, __o__, __m__, __x__ ) \
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \
   __typeof__(__m__) __v__ = (__m__); \
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \
   __atomic_flag_wait_explicit__( __g__, __x__ ); \
   __typeof__((__a__)->__f__) __r__ = *__p__; \
   *__p__ __o__ __v__; \
   atomic_flag_clear_explicit( __g__, __x__ ); \
   __r__; })

#define _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ) \
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \
   __typeof__(__e__) __q__ = (__e__); \
   __typeof__(__m__) __v__ = (__m__); \
   bool __r__; \
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \
   __atomic_flag_wait_explicit__( __g__, __x__ ); \
   __typeof__((__a__)->__f__) __t__ = *__p__; \
   if ( __t__ == *__q__ ) { *__p__ = __v__; __r__ = true; } \
   else { *__q__ = __t__; __r__ = false; } \
   atomic_flag_clear_explicit( __g__, __x__ ); \
   __r__; })

#define _ATOMIC_FENCE_( __a__, __x__ ) \
({ volatile __typeof__((__a__)->__f__)* __p__ = &((__a__)->__f__); \
   volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ ); \
   atomic_flag_fence( __g__, __x__ ); \
   })


#define ATOMIC_INTEGRAL_LOCK_FREE 0
#define ATOMIC_ADDRESS_LOCK_FREE 0


typedef struct atomic_bool
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( bool, memory_order = memory_order_seq_cst ) volatile;
    bool load( memory_order = memory_order_seq_cst ) volatile;
    bool swap( bool, memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap ( bool&, bool, memory_order, memory_order ) volatile;
    bool compare_swap ( bool&, bool,
                        memory_order = memory_order_seq_cst) volatile;
    void fence( memory_order ) const volatile;

    CPP0X( atomic_bool() = delete; )
    CPP0X( constexpr explicit atomic_bool( bool __v__ ) : __f__( __v__ ) { } )
    CPP0X( atomic_bool( const atomic_bool& ) = delete; )
    atomic_bool& operator =( const atomic_bool& ) CPP0X(=delete);

    bool operator =( bool __v__ ) volatile
    { store( __v__ ); return __v__; }

    friend void atomic_store_explicit( volatile atomic_bool*, bool,
                                       memory_order );
    friend bool atomic_load_explicit( volatile atomic_bool*, memory_order );
    friend bool atomic_swap_explicit( volatile atomic_bool*, bool,
                                      memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_bool*, bool*, bool,
                                              memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_bool*, memory_order );

CPP0X(private:)
#endif
    bool __f__;
} atomic_bool;


typedef struct atomic_address
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( void*, memory_order = memory_order_seq_cst ) volatile;
    void* load( memory_order = memory_order_seq_cst ) volatile;
    void* swap( void*, memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( void*&, void*, memory_order, memory_order ) volatile;
    bool compare_swap( void*&, void*,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    void* fetch_add( ptrdiff_t, memory_order = memory_order_seq_cst ) volatile;
    void* fetch_sub( ptrdiff_t, memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_address() = default; )
    CPP0X( constexpr explicit atomic_address( void* __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_address( const atomic_address& ) = delete; )
    atomic_address& operator =( const atomic_address & ) CPP0X(=delete);

    void* operator =( void* __v__ ) volatile
    { store( __v__ ); return __v__; }

    void* operator +=( ptrdiff_t __v__ ) volatile
    { return fetch_add( __v__ ); }

    void* operator -=( ptrdiff_t __v__ ) volatile
    { return fetch_sub( __v__ ); }

    friend void atomic_store_explicit( volatile atomic_address*, void*,
                                       memory_order );
    friend void* atomic_load_explicit( volatile atomic_address*, memory_order );
    friend void* atomic_swap_explicit( volatile atomic_address*, void*,
                                       memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_address*,
                              void**, void*, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_address*, memory_order );
    friend void* atomic_fetch_add_explicit( volatile atomic_address*, ptrdiff_t,
                                            memory_order );
    friend void* atomic_fetch_sub_explicit( volatile atomic_address*, ptrdiff_t,
                                            memory_order );

CPP0X(private:)
#endif
    void* __f__;
} atomic_address;


typedef struct atomic_char
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( char,
                memory_order = memory_order_seq_cst ) volatile;
    char load( memory_order = memory_order_seq_cst ) volatile;
    char swap( char,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( char&, char,
                       memory_order, memory_order ) volatile;
    bool compare_swap( char&, char,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    char fetch_add( char,
                           memory_order = memory_order_seq_cst ) volatile;
    char fetch_sub( char,
                           memory_order = memory_order_seq_cst ) volatile;
    char fetch_and( char,
                           memory_order = memory_order_seq_cst ) volatile;
    char fetch_or( char,
                           memory_order = memory_order_seq_cst ) volatile;
    char fetch_xor( char,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_char() = default; )
    CPP0X( constexpr atomic_char( char __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_char( const atomic_char& ) = delete; )
    atomic_char& operator =( const atomic_char& ) CPP0X(=delete);

    char operator =( char __v__ ) volatile
    { store( __v__ ); return __v__; }

    char operator ++( int ) volatile
    { return fetch_add( 1 ); }

    char operator --( int ) volatile
    { return fetch_sub( 1 ); }

    char operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    char operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    char operator +=( char __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    char operator -=( char __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    char operator &=( char __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    char operator |=( char __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    char operator ^=( char __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_char*, char,
                                       memory_order );
    friend char atomic_load_explicit( volatile atomic_char*,
                                             memory_order );
    friend char atomic_swap_explicit( volatile atomic_char*,
                                             char, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_char*,
                      char*, char, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_char*, memory_order );
    friend char atomic_fetch_add_explicit( volatile atomic_char*,
                                                  char, memory_order );
    friend char atomic_fetch_sub_explicit( volatile atomic_char*,
                                                  char, memory_order );
    friend char atomic_fetch_and_explicit( volatile atomic_char*,
                                                  char, memory_order );
    friend char atomic_fetch_or_explicit(  volatile atomic_char*,
                                                  char, memory_order );
    friend char atomic_fetch_xor_explicit( volatile atomic_char*,
                                                  char, memory_order );

CPP0X(private:)
#endif
    char __f__;
} atomic_char;


typedef struct atomic_schar
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( signed char,
                memory_order = memory_order_seq_cst ) volatile;
    signed char load( memory_order = memory_order_seq_cst ) volatile;
    signed char swap( signed char,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( signed char&, signed char,
                       memory_order, memory_order ) volatile;
    bool compare_swap( signed char&, signed char,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    signed char fetch_add( signed char,
                           memory_order = memory_order_seq_cst ) volatile;
    signed char fetch_sub( signed char,
                           memory_order = memory_order_seq_cst ) volatile;
    signed char fetch_and( signed char,
                           memory_order = memory_order_seq_cst ) volatile;
    signed char fetch_or( signed char,
                           memory_order = memory_order_seq_cst ) volatile;
    signed char fetch_xor( signed char,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_schar() = default; )
    CPP0X( constexpr atomic_schar( signed char __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_schar( const atomic_schar& ) = delete; )
    atomic_schar& operator =( const atomic_schar& ) CPP0X(=delete);

    signed char operator =( signed char __v__ ) volatile
    { store( __v__ ); return __v__; }

    signed char operator ++( int ) volatile
    { return fetch_add( 1 ); }

    signed char operator --( int ) volatile
    { return fetch_sub( 1 ); }

    signed char operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    signed char operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    signed char operator +=( signed char __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    signed char operator -=( signed char __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    signed char operator &=( signed char __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    signed char operator |=( signed char __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    signed char operator ^=( signed char __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_schar*, signed char,
                                       memory_order );
    friend signed char atomic_load_explicit( volatile atomic_schar*,
                                             memory_order );
    friend signed char atomic_swap_explicit( volatile atomic_schar*,
                                             signed char, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_schar*,
                      signed char*, signed char, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_schar*, memory_order );
    friend signed char atomic_fetch_add_explicit( volatile atomic_schar*,
                                                  signed char, memory_order );
    friend signed char atomic_fetch_sub_explicit( volatile atomic_schar*,
                                                  signed char, memory_order );
    friend signed char atomic_fetch_and_explicit( volatile atomic_schar*,
                                                  signed char, memory_order );
    friend signed char atomic_fetch_or_explicit(  volatile atomic_schar*,
                                                  signed char, memory_order );
    friend signed char atomic_fetch_xor_explicit( volatile atomic_schar*,
                                                  signed char, memory_order );

CPP0X(private:)
#endif
    signed char __f__;
} atomic_schar;


typedef struct atomic_uchar
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( unsigned char,
                memory_order = memory_order_seq_cst ) volatile;
    unsigned char load( memory_order = memory_order_seq_cst ) volatile;
    unsigned char swap( unsigned char,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( unsigned char&, unsigned char,
                       memory_order, memory_order ) volatile;
    bool compare_swap( unsigned char&, unsigned char,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    unsigned char fetch_add( unsigned char,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned char fetch_sub( unsigned char,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned char fetch_and( unsigned char,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned char fetch_or( unsigned char,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned char fetch_xor( unsigned char,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_uchar() = default; )
    CPP0X( constexpr atomic_uchar( unsigned char __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_uchar( const atomic_uchar& ) = delete; )
    atomic_uchar& operator =( const atomic_uchar& ) CPP0X(=delete);

    unsigned char operator =( unsigned char __v__ ) volatile
    { store( __v__ ); return __v__; }

    unsigned char operator ++( int ) volatile
    { return fetch_add( 1 ); }

    unsigned char operator --( int ) volatile
    { return fetch_sub( 1 ); }

    unsigned char operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    unsigned char operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    unsigned char operator +=( unsigned char __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    unsigned char operator -=( unsigned char __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    unsigned char operator &=( unsigned char __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    unsigned char operator |=( unsigned char __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    unsigned char operator ^=( unsigned char __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_uchar*, unsigned char,
                                       memory_order );
    friend unsigned char atomic_load_explicit( volatile atomic_uchar*,
                                             memory_order );
    friend unsigned char atomic_swap_explicit( volatile atomic_uchar*,
                                             unsigned char, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_uchar*,
                      unsigned char*, unsigned char, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_uchar*, memory_order );
    friend unsigned char atomic_fetch_add_explicit( volatile atomic_uchar*,
                                                  unsigned char, memory_order );
    friend unsigned char atomic_fetch_sub_explicit( volatile atomic_uchar*,
                                                  unsigned char, memory_order );
    friend unsigned char atomic_fetch_and_explicit( volatile atomic_uchar*,
                                                  unsigned char, memory_order );
    friend unsigned char atomic_fetch_or_explicit(  volatile atomic_uchar*,
                                                  unsigned char, memory_order );
    friend unsigned char atomic_fetch_xor_explicit( volatile atomic_uchar*,
                                                  unsigned char, memory_order );

CPP0X(private:)
#endif
    unsigned char __f__;
} atomic_uchar;


typedef struct atomic_short
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( short,
                memory_order = memory_order_seq_cst ) volatile;
    short load( memory_order = memory_order_seq_cst ) volatile;
    short swap( short,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( short&, short,
                       memory_order, memory_order ) volatile;
    bool compare_swap( short&, short,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    short fetch_add( short,
                           memory_order = memory_order_seq_cst ) volatile;
    short fetch_sub( short,
                           memory_order = memory_order_seq_cst ) volatile;
    short fetch_and( short,
                           memory_order = memory_order_seq_cst ) volatile;
    short fetch_or( short,
                           memory_order = memory_order_seq_cst ) volatile;
    short fetch_xor( short,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_short() = default; )
    CPP0X( constexpr atomic_short( short __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_short( const atomic_short& ) = delete; )
    atomic_short& operator =( const atomic_short& ) CPP0X(=delete);

    short operator =( short __v__ ) volatile
    { store( __v__ ); return __v__; }

    short operator ++( int ) volatile
    { return fetch_add( 1 ); }

    short operator --( int ) volatile
    { return fetch_sub( 1 ); }

    short operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    short operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    short operator +=( short __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    short operator -=( short __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    short operator &=( short __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    short operator |=( short __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    short operator ^=( short __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_short*, short,
                                       memory_order );
    friend short atomic_load_explicit( volatile atomic_short*,
                                             memory_order );
    friend short atomic_swap_explicit( volatile atomic_short*,
                                             short, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_short*,
                      short*, short, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_short*, memory_order );
    friend short atomic_fetch_add_explicit( volatile atomic_short*,
                                                  short, memory_order );
    friend short atomic_fetch_sub_explicit( volatile atomic_short*,
                                                  short, memory_order );
    friend short atomic_fetch_and_explicit( volatile atomic_short*,
                                                  short, memory_order );
    friend short atomic_fetch_or_explicit(  volatile atomic_short*,
                                                  short, memory_order );
    friend short atomic_fetch_xor_explicit( volatile atomic_short*,
                                                  short, memory_order );

CPP0X(private:)
#endif
    short __f__;
} atomic_short;


typedef struct atomic_ushort
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( unsigned short,
                memory_order = memory_order_seq_cst ) volatile;
    unsigned short load( memory_order = memory_order_seq_cst ) volatile;
    unsigned short swap( unsigned short,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( unsigned short&, unsigned short,
                       memory_order, memory_order ) volatile;
    bool compare_swap( unsigned short&, unsigned short,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    unsigned short fetch_add( unsigned short,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned short fetch_sub( unsigned short,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned short fetch_and( unsigned short,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned short fetch_or( unsigned short,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned short fetch_xor( unsigned short,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_ushort() = default; )
    CPP0X( constexpr atomic_ushort( unsigned short __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_ushort( const atomic_ushort& ) = delete; )
    atomic_ushort& operator =( const atomic_ushort& ) CPP0X(=delete);

    unsigned short operator =( unsigned short __v__ ) volatile
    { store( __v__ ); return __v__; }

    unsigned short operator ++( int ) volatile
    { return fetch_add( 1 ); }

    unsigned short operator --( int ) volatile
    { return fetch_sub( 1 ); }

    unsigned short operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    unsigned short operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    unsigned short operator +=( unsigned short __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    unsigned short operator -=( unsigned short __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    unsigned short operator &=( unsigned short __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    unsigned short operator |=( unsigned short __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    unsigned short operator ^=( unsigned short __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_ushort*, unsigned short,
                                       memory_order );
    friend unsigned short atomic_load_explicit( volatile atomic_ushort*,
                                             memory_order );
    friend unsigned short atomic_swap_explicit( volatile atomic_ushort*,
                                             unsigned short, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_ushort*,
                      unsigned short*, unsigned short, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_ushort*, memory_order );
    friend unsigned short atomic_fetch_add_explicit( volatile atomic_ushort*,
                                                  unsigned short, memory_order );
    friend unsigned short atomic_fetch_sub_explicit( volatile atomic_ushort*,
                                                  unsigned short, memory_order );
    friend unsigned short atomic_fetch_and_explicit( volatile atomic_ushort*,
                                                  unsigned short, memory_order );
    friend unsigned short atomic_fetch_or_explicit(  volatile atomic_ushort*,
                                                  unsigned short, memory_order );
    friend unsigned short atomic_fetch_xor_explicit( volatile atomic_ushort*,
                                                  unsigned short, memory_order );

CPP0X(private:)
#endif
    unsigned short __f__;
} atomic_ushort;


typedef struct atomic_int
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( int,
                memory_order = memory_order_seq_cst ) volatile;
    int load( memory_order = memory_order_seq_cst ) volatile;
    int swap( int,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( int&, int,
                       memory_order, memory_order ) volatile;
    bool compare_swap( int&, int,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    int fetch_add( int,
                           memory_order = memory_order_seq_cst ) volatile;
    int fetch_sub( int,
                           memory_order = memory_order_seq_cst ) volatile;
    int fetch_and( int,
                           memory_order = memory_order_seq_cst ) volatile;
    int fetch_or( int,
                           memory_order = memory_order_seq_cst ) volatile;
    int fetch_xor( int,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_int() = default; )
    CPP0X( constexpr atomic_int( int __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_int( const atomic_int& ) = delete; )
    atomic_int& operator =( const atomic_int& ) CPP0X(=delete);

    int operator =( int __v__ ) volatile
    { store( __v__ ); return __v__; }

    int operator ++( int ) volatile
    { return fetch_add( 1 ); }

    int operator --( int ) volatile
    { return fetch_sub( 1 ); }

    int operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    int operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    int operator +=( int __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    int operator -=( int __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    int operator &=( int __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    int operator |=( int __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    int operator ^=( int __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_int*, int,
                                       memory_order );
    friend int atomic_load_explicit( volatile atomic_int*,
                                             memory_order );
    friend int atomic_swap_explicit( volatile atomic_int*,
                                             int, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_int*,
                      int*, int, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_int*, memory_order );
    friend int atomic_fetch_add_explicit( volatile atomic_int*,
                                                  int, memory_order );
    friend int atomic_fetch_sub_explicit( volatile atomic_int*,
                                                  int, memory_order );
    friend int atomic_fetch_and_explicit( volatile atomic_int*,
                                                  int, memory_order );
    friend int atomic_fetch_or_explicit(  volatile atomic_int*,
                                                  int, memory_order );
    friend int atomic_fetch_xor_explicit( volatile atomic_int*,
                                                  int, memory_order );

CPP0X(private:)
#endif
    int __f__;
} atomic_int;


typedef struct atomic_uint
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( unsigned int,
                memory_order = memory_order_seq_cst ) volatile;
    unsigned int load( memory_order = memory_order_seq_cst ) volatile;
    unsigned int swap( unsigned int,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( unsigned int&, unsigned int,
                       memory_order, memory_order ) volatile;
    bool compare_swap( unsigned int&, unsigned int,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    unsigned int fetch_add( unsigned int,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned int fetch_sub( unsigned int,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned int fetch_and( unsigned int,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned int fetch_or( unsigned int,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned int fetch_xor( unsigned int,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_uint() = default; )
    CPP0X( constexpr atomic_uint( unsigned int __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_uint( const atomic_uint& ) = delete; )
    atomic_uint& operator =( const atomic_uint& ) CPP0X(=delete);

    unsigned int operator =( unsigned int __v__ ) volatile
    { store( __v__ ); return __v__; }

    unsigned int operator ++( int ) volatile
    { return fetch_add( 1 ); }

    unsigned int operator --( int ) volatile
    { return fetch_sub( 1 ); }

    unsigned int operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    unsigned int operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    unsigned int operator +=( unsigned int __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    unsigned int operator -=( unsigned int __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    unsigned int operator &=( unsigned int __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    unsigned int operator |=( unsigned int __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    unsigned int operator ^=( unsigned int __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_uint*, unsigned int,
                                       memory_order );
    friend unsigned int atomic_load_explicit( volatile atomic_uint*,
                                             memory_order );
    friend unsigned int atomic_swap_explicit( volatile atomic_uint*,
                                             unsigned int, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_uint*,
                      unsigned int*, unsigned int, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_uint*, memory_order );
    friend unsigned int atomic_fetch_add_explicit( volatile atomic_uint*,
                                                  unsigned int, memory_order );
    friend unsigned int atomic_fetch_sub_explicit( volatile atomic_uint*,
                                                  unsigned int, memory_order );
    friend unsigned int atomic_fetch_and_explicit( volatile atomic_uint*,
                                                  unsigned int, memory_order );
    friend unsigned int atomic_fetch_or_explicit(  volatile atomic_uint*,
                                                  unsigned int, memory_order );
    friend unsigned int atomic_fetch_xor_explicit( volatile atomic_uint*,
                                                  unsigned int, memory_order );

CPP0X(private:)
#endif
    unsigned int __f__;
} atomic_uint;


typedef struct atomic_long
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( long,
                memory_order = memory_order_seq_cst ) volatile;
    long load( memory_order = memory_order_seq_cst ) volatile;
    long swap( long,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( long&, long,
                       memory_order, memory_order ) volatile;
    bool compare_swap( long&, long,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    long fetch_add( long,
                           memory_order = memory_order_seq_cst ) volatile;
    long fetch_sub( long,
                           memory_order = memory_order_seq_cst ) volatile;
    long fetch_and( long,
                           memory_order = memory_order_seq_cst ) volatile;
    long fetch_or( long,
                           memory_order = memory_order_seq_cst ) volatile;
    long fetch_xor( long,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_long() = default; )
    CPP0X( constexpr atomic_long( long __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_long( const atomic_long& ) = delete; )
    atomic_long& operator =( const atomic_long& ) CPP0X(=delete);

    long operator =( long __v__ ) volatile
    { store( __v__ ); return __v__; }

    long operator ++( int ) volatile
    { return fetch_add( 1 ); }

    long operator --( int ) volatile
    { return fetch_sub( 1 ); }

    long operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    long operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    long operator +=( long __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    long operator -=( long __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    long operator &=( long __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    long operator |=( long __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    long operator ^=( long __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_long*, long,
                                       memory_order );
    friend long atomic_load_explicit( volatile atomic_long*,
                                             memory_order );
    friend long atomic_swap_explicit( volatile atomic_long*,
                                             long, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_long*,
                      long*, long, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_long*, memory_order );
    friend long atomic_fetch_add_explicit( volatile atomic_long*,
                                                  long, memory_order );
    friend long atomic_fetch_sub_explicit( volatile atomic_long*,
                                                  long, memory_order );
    friend long atomic_fetch_and_explicit( volatile atomic_long*,
                                                  long, memory_order );
    friend long atomic_fetch_or_explicit(  volatile atomic_long*,
                                                  long, memory_order );
    friend long atomic_fetch_xor_explicit( volatile atomic_long*,
                                                  long, memory_order );

CPP0X(private:)
#endif
    long __f__;
} atomic_long;


typedef struct atomic_ulong
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( unsigned long,
                memory_order = memory_order_seq_cst ) volatile;
    unsigned long load( memory_order = memory_order_seq_cst ) volatile;
    unsigned long swap( unsigned long,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( unsigned long&, unsigned long,
                       memory_order, memory_order ) volatile;
    bool compare_swap( unsigned long&, unsigned long,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    unsigned long fetch_add( unsigned long,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned long fetch_sub( unsigned long,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned long fetch_and( unsigned long,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned long fetch_or( unsigned long,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned long fetch_xor( unsigned long,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_ulong() = default; )
    CPP0X( constexpr atomic_ulong( unsigned long __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_ulong( const atomic_ulong& ) = delete; )
    atomic_ulong& operator =( const atomic_ulong& ) CPP0X(=delete);

    unsigned long operator =( unsigned long __v__ ) volatile
    { store( __v__ ); return __v__; }

    unsigned long operator ++( int ) volatile
    { return fetch_add( 1 ); }

    unsigned long operator --( int ) volatile
    { return fetch_sub( 1 ); }

    unsigned long operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    unsigned long operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    unsigned long operator +=( unsigned long __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    unsigned long operator -=( unsigned long __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    unsigned long operator &=( unsigned long __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    unsigned long operator |=( unsigned long __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    unsigned long operator ^=( unsigned long __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_ulong*, unsigned long,
                                       memory_order );
    friend unsigned long atomic_load_explicit( volatile atomic_ulong*,
                                             memory_order );
    friend unsigned long atomic_swap_explicit( volatile atomic_ulong*,
                                             unsigned long, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_ulong*,
                      unsigned long*, unsigned long, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_ulong*, memory_order );
    friend unsigned long atomic_fetch_add_explicit( volatile atomic_ulong*,
                                                  unsigned long, memory_order );
    friend unsigned long atomic_fetch_sub_explicit( volatile atomic_ulong*,
                                                  unsigned long, memory_order );
    friend unsigned long atomic_fetch_and_explicit( volatile atomic_ulong*,
                                                  unsigned long, memory_order );
    friend unsigned long atomic_fetch_or_explicit(  volatile atomic_ulong*,
                                                  unsigned long, memory_order );
    friend unsigned long atomic_fetch_xor_explicit( volatile atomic_ulong*,
                                                  unsigned long, memory_order );

CPP0X(private:)
#endif
    unsigned long __f__;
} atomic_ulong;


typedef struct atomic_llong
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( long long,
                memory_order = memory_order_seq_cst ) volatile;
    long long load( memory_order = memory_order_seq_cst ) volatile;
    long long swap( long long,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( long long&, long long,
                       memory_order, memory_order ) volatile;
    bool compare_swap( long long&, long long,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    long long fetch_add( long long,
                           memory_order = memory_order_seq_cst ) volatile;
    long long fetch_sub( long long,
                           memory_order = memory_order_seq_cst ) volatile;
    long long fetch_and( long long,
                           memory_order = memory_order_seq_cst ) volatile;
    long long fetch_or( long long,
                           memory_order = memory_order_seq_cst ) volatile;
    long long fetch_xor( long long,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_llong() = default; )
    CPP0X( constexpr atomic_llong( long long __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_llong( const atomic_llong& ) = delete; )
    atomic_llong& operator =( const atomic_llong& ) CPP0X(=delete);

    long long operator =( long long __v__ ) volatile
    { store( __v__ ); return __v__; }

    long long operator ++( int ) volatile
    { return fetch_add( 1 ); }

    long long operator --( int ) volatile
    { return fetch_sub( 1 ); }

    long long operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    long long operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    long long operator +=( long long __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    long long operator -=( long long __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    long long operator &=( long long __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    long long operator |=( long long __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    long long operator ^=( long long __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_llong*, long long,
                                       memory_order );
    friend long long atomic_load_explicit( volatile atomic_llong*,
                                             memory_order );
    friend long long atomic_swap_explicit( volatile atomic_llong*,
                                             long long, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_llong*,
                      long long*, long long, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_llong*, memory_order );
    friend long long atomic_fetch_add_explicit( volatile atomic_llong*,
                                                  long long, memory_order );
    friend long long atomic_fetch_sub_explicit( volatile atomic_llong*,
                                                  long long, memory_order );
    friend long long atomic_fetch_and_explicit( volatile atomic_llong*,
                                                  long long, memory_order );
    friend long long atomic_fetch_or_explicit(  volatile atomic_llong*,
                                                  long long, memory_order );
    friend long long atomic_fetch_xor_explicit( volatile atomic_llong*,
                                                  long long, memory_order );

CPP0X(private:)
#endif
    long long __f__;
} atomic_llong;


typedef struct atomic_ullong
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( unsigned long long,
                memory_order = memory_order_seq_cst ) volatile;
    unsigned long long load( memory_order = memory_order_seq_cst ) volatile;
    unsigned long long swap( unsigned long long,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( unsigned long long&, unsigned long long,
                       memory_order, memory_order ) volatile;
    bool compare_swap( unsigned long long&, unsigned long long,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    unsigned long long fetch_add( unsigned long long,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned long long fetch_sub( unsigned long long,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned long long fetch_and( unsigned long long,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned long long fetch_or( unsigned long long,
                           memory_order = memory_order_seq_cst ) volatile;
    unsigned long long fetch_xor( unsigned long long,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_ullong() = default; )
    CPP0X( constexpr atomic_ullong( unsigned long long __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_ullong( const atomic_ullong& ) = delete; )
    atomic_ullong& operator =( const atomic_ullong& ) CPP0X(=delete);

    unsigned long long operator =( unsigned long long __v__ ) volatile
    { store( __v__ ); return __v__; }

    unsigned long long operator ++( int ) volatile
    { return fetch_add( 1 ); }

    unsigned long long operator --( int ) volatile
    { return fetch_sub( 1 ); }

    unsigned long long operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    unsigned long long operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    unsigned long long operator +=( unsigned long long __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    unsigned long long operator -=( unsigned long long __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    unsigned long long operator &=( unsigned long long __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    unsigned long long operator |=( unsigned long long __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    unsigned long long operator ^=( unsigned long long __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_ullong*, unsigned long long,
                                       memory_order );
    friend unsigned long long atomic_load_explicit( volatile atomic_ullong*,
                                             memory_order );
    friend unsigned long long atomic_swap_explicit( volatile atomic_ullong*,
                                             unsigned long long, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_ullong*,
                      unsigned long long*, unsigned long long, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_ullong*, memory_order );
    friend unsigned long long atomic_fetch_add_explicit( volatile atomic_ullong*,
                                                  unsigned long long, memory_order );
    friend unsigned long long atomic_fetch_sub_explicit( volatile atomic_ullong*,
                                                  unsigned long long, memory_order );
    friend unsigned long long atomic_fetch_and_explicit( volatile atomic_ullong*,
                                                  unsigned long long, memory_order );
    friend unsigned long long atomic_fetch_or_explicit(  volatile atomic_ullong*,
                                                  unsigned long long, memory_order );
    friend unsigned long long atomic_fetch_xor_explicit( volatile atomic_ullong*,
                                                  unsigned long long, memory_order );

CPP0X(private:)
#endif
    unsigned long long __f__;
} atomic_ullong;


typedef atomic_schar atomic_int_least8_t;
typedef atomic_uchar atomic_uint_least8_t;
typedef atomic_short atomic_int_least16_t;
typedef atomic_ushort atomic_uint_least16_t;
typedef atomic_int atomic_int_least32_t;
typedef atomic_uint atomic_uint_least32_t;
typedef atomic_llong atomic_int_least64_t;
typedef atomic_ullong atomic_uint_least64_t;

typedef atomic_schar atomic_int_fast8_t;
typedef atomic_uchar atomic_uint_fast8_t;
typedef atomic_short atomic_int_fast16_t;
typedef atomic_ushort atomic_uint_fast16_t;
typedef atomic_int atomic_int_fast32_t;
typedef atomic_uint atomic_uint_fast32_t;
typedef atomic_llong atomic_int_fast64_t;
typedef atomic_ullong atomic_uint_fast64_t;

typedef atomic_long atomic_intptr_t;
typedef atomic_ulong atomic_uintptr_t;

typedef atomic_long atomic_ssize_t;
typedef atomic_ulong atomic_size_t;

typedef atomic_long atomic_ptrdiff_t;

typedef atomic_llong atomic_intmax_t;
typedef atomic_ullong atomic_uintmax_t;


#ifdef __cplusplus


typedef struct atomic_wchar_t
{
#ifdef __cplusplus
    bool is_lock_free() const volatile;
    void store( wchar_t, memory_order = memory_order_seq_cst ) volatile;
    wchar_t load( memory_order = memory_order_seq_cst ) volatile;
    wchar_t swap( wchar_t,
                      memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( wchar_t&, wchar_t,
                       memory_order, memory_order ) volatile;
    bool compare_swap( wchar_t&, wchar_t,
                       memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;
    wchar_t fetch_add( wchar_t,
                           memory_order = memory_order_seq_cst ) volatile;
    wchar_t fetch_sub( wchar_t,
                           memory_order = memory_order_seq_cst ) volatile;
    wchar_t fetch_and( wchar_t,
                           memory_order = memory_order_seq_cst ) volatile;
    wchar_t fetch_or( wchar_t,
                           memory_order = memory_order_seq_cst ) volatile;
    wchar_t fetch_xor( wchar_t,
                           memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic_wchar_t() = default; )
    CPP0X( constexpr atomic_wchar_t( wchar_t __v__ ) : __f__( __v__) { } )
    CPP0X( atomic_wchar_t( const atomic_wchar_t& ) = delete; )
    atomic_wchar_t& operator =( const atomic_wchar_t& ) CPP0X(=delete);

    wchar_t operator =( wchar_t __v__ ) volatile
    { store( __v__ ); return __v__; }

    wchar_t operator ++( int ) volatile
    { return fetch_add( 1 ); }

    wchar_t operator --( int ) volatile
    { return fetch_sub( 1 ); }

    wchar_t operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    wchar_t operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    wchar_t operator +=( wchar_t __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    wchar_t operator -=( wchar_t __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }

    wchar_t operator &=( wchar_t __v__ ) volatile
    { return fetch_and( __v__ ) & __v__; }

    wchar_t operator |=( wchar_t __v__ ) volatile
    { return fetch_or( __v__ ) | __v__; }

    wchar_t operator ^=( wchar_t __v__ ) volatile
    { return fetch_xor( __v__ ) ^ __v__; }

    friend void atomic_store_explicit( volatile atomic_wchar_t*, wchar_t,
                                       memory_order );
    friend wchar_t atomic_load_explicit( volatile atomic_wchar_t*,
                                             memory_order );
    friend wchar_t atomic_swap_explicit( volatile atomic_wchar_t*,
                                             wchar_t, memory_order );
    friend bool atomic_compare_swap_explicit( volatile atomic_wchar_t*,
                    wchar_t*, wchar_t, memory_order, memory_order );
    friend void atomic_fence( const volatile atomic_wchar_t*, memory_order );
    friend wchar_t atomic_fetch_add_explicit( volatile atomic_wchar_t*,
                                                  wchar_t, memory_order );
    friend wchar_t atomic_fetch_sub_explicit( volatile atomic_wchar_t*,
                                                  wchar_t, memory_order );
    friend wchar_t atomic_fetch_and_explicit( volatile atomic_wchar_t*,
                                                  wchar_t, memory_order );
    friend wchar_t atomic_fetch_or_explicit( volatile atomic_wchar_t*,
                                                  wchar_t, memory_order );
    friend wchar_t atomic_fetch_xor_explicit( volatile atomic_wchar_t*,
                                                  wchar_t, memory_order );

CPP0X(private:)
#endif
    wchar_t __f__;
} atomic_wchar_t;


#else

typedef atomic_int_least16_t atomic_char16_t;
typedef atomic_int_least32_t atomic_char32_t;
typedef atomic_int_least32_t atomic_wchar_t;

#endif


#ifdef __cplusplus

template< typename T >
struct atomic
{
#ifdef __cplusplus

    bool is_lock_free() const volatile;
    void store( T, memory_order = memory_order_seq_cst ) volatile;
    T load( memory_order = memory_order_seq_cst ) volatile;
    T swap( T __v__, memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( T&, T, memory_order, memory_order ) volatile;
    bool compare_swap( T&, T, memory_order = memory_order_seq_cst ) volatile;
    void fence( memory_order ) const volatile;

    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( T __v__ ) : __f__( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    T operator =( T __v__ ) volatile
    { store( __v__ ); return __v__; }

CPP0X(private:)
#endif
    T __f__;
};

#endif

#ifdef __cplusplus

template<typename T> struct atomic< T* > : atomic_address
{
    T* load( memory_order = memory_order_seq_cst ) volatile;
    T* swap( T*, memory_order = memory_order_seq_cst ) volatile;
    bool compare_swap( T*&, T*, memory_order, memory_order ) volatile;
    bool compare_swap( T*&, T*,
                       memory_order = memory_order_seq_cst ) volatile;
    T* fetch_add( ptrdiff_t, memory_order = memory_order_seq_cst ) volatile;
    T* fetch_sub( ptrdiff_t, memory_order = memory_order_seq_cst ) volatile;

    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( T __v__ ) : atomic_address( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    T* operator =( T* __v__ ) volatile
    { store( __v__ ); return __v__; }

    T* operator ++( int ) volatile
    { return fetch_add( 1 ); }

    T* operator --( int ) volatile
    { return fetch_sub( 1 ); }

    T* operator ++() volatile
    { return fetch_add( 1 ) + 1; }

    T* operator --() volatile
    { return fetch_sub( 1 ) - 1; }

    T* operator +=( T* __v__ ) volatile
    { return fetch_add( __v__ ) + __v__; }

    T* operator -=( T* __v__ ) volatile
    { return fetch_sub( __v__ ) - __v__; }
};

#endif

#ifdef __cplusplus


template<> struct atomic< bool > : atomic_bool
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( bool __v__ )
    : atomic_bool( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    bool operator =( bool __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< void* > : atomic_address
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( void* __v__ )
    : atomic_address( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    void* operator =( void* __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< char > : atomic_char
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( char __v__ )
    : atomic_char( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    char operator =( char __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< signed char > : atomic_schar
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( signed char __v__ )
    : atomic_schar( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    signed char operator =( signed char __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< unsigned char > : atomic_uchar
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( unsigned char __v__ )
    : atomic_uchar( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    unsigned char operator =( unsigned char __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< short > : atomic_short
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( short __v__ )
    : atomic_short( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    short operator =( short __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< unsigned short > : atomic_ushort
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( unsigned short __v__ )
    : atomic_ushort( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    unsigned short operator =( unsigned short __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< int > : atomic_int
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( int __v__ )
    : atomic_int( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    int operator =( int __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< unsigned int > : atomic_uint
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( unsigned int __v__ )
    : atomic_uint( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    unsigned int operator =( unsigned int __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< long > : atomic_long
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( long __v__ )
    : atomic_long( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    long operator =( long __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< unsigned long > : atomic_ulong
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( unsigned long __v__ )
    : atomic_ulong( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    unsigned long operator =( unsigned long __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< long long > : atomic_llong
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( long long __v__ )
    : atomic_llong( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    long long operator =( long long __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< unsigned long long > : atomic_ullong
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( unsigned long long __v__ )
    : atomic_ullong( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    unsigned long long operator =( unsigned long long __v__ ) volatile
    { store( __v__ ); return __v__; }
};


template<> struct atomic< wchar_t > : atomic_wchar_t
{
    CPP0X( atomic() = default; )
    CPP0X( constexpr explicit atomic( wchar_t __v__ )
    : atomic_wchar_t( __v__ ) { } )
    CPP0X( atomic( const atomic& ) = delete; )
    atomic& operator =( const atomic& ) CPP0X(=delete);

    wchar_t operator =( wchar_t __v__ ) volatile
    { store( __v__ ); return __v__; }
};


#endif


#ifdef __cplusplus


inline bool atomic_is_lock_free( const volatile atomic_bool* __a__ )
{ return false; }

inline bool atomic_load_explicit
( volatile atomic_bool* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline bool atomic_load( volatile atomic_bool* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_bool* __a__, bool __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_bool* __a__, bool __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_swap_explicit
( volatile atomic_bool* __a__, bool __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline bool atomic_swap
( volatile atomic_bool* __a__, bool __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_bool* __a__, bool* __e__, bool __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_bool* __a__, bool* __e__, bool __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_bool* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_address* __a__ )
{ return false; }

inline void* atomic_load_explicit
( volatile atomic_address* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline void* atomic_load( volatile atomic_address* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_address* __a__, void* __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_address* __a__, void* __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline void* atomic_swap_explicit
( volatile atomic_address* __a__, void* __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline void* atomic_swap
( volatile atomic_address* __a__, void* __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_address* __a__, void** __e__, void* __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_address* __a__, void** __e__, void* __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_address* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_char* __a__ )
{ return false; }

inline char atomic_load_explicit
( volatile atomic_char* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline char atomic_load( volatile atomic_char* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_char* __a__, char __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_char* __a__, char __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline char atomic_swap_explicit
( volatile atomic_char* __a__, char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline char atomic_swap
( volatile atomic_char* __a__, char __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_char* __a__, char* __e__, char __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_char* __a__, char* __e__, char __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_char* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_schar* __a__ )
{ return false; }

inline signed char atomic_load_explicit
( volatile atomic_schar* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline signed char atomic_load( volatile atomic_schar* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_schar* __a__, signed char __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_schar* __a__, signed char __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline signed char atomic_swap_explicit
( volatile atomic_schar* __a__, signed char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline signed char atomic_swap
( volatile atomic_schar* __a__, signed char __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_schar* __a__, signed char* __e__, signed char __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_schar* __a__, signed char* __e__, signed char __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_schar* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_uchar* __a__ )
{ return false; }

inline unsigned char atomic_load_explicit
( volatile atomic_uchar* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline unsigned char atomic_load( volatile atomic_uchar* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_uchar* __a__, unsigned char __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_uchar* __a__, unsigned char __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline unsigned char atomic_swap_explicit
( volatile atomic_uchar* __a__, unsigned char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline unsigned char atomic_swap
( volatile atomic_uchar* __a__, unsigned char __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_uchar* __a__, unsigned char* __e__, unsigned char __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_uchar* __a__, unsigned char* __e__, unsigned char __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_uchar* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_short* __a__ )
{ return false; }

inline short atomic_load_explicit
( volatile atomic_short* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline short atomic_load( volatile atomic_short* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_short* __a__, short __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_short* __a__, short __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline short atomic_swap_explicit
( volatile atomic_short* __a__, short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline short atomic_swap
( volatile atomic_short* __a__, short __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_short* __a__, short* __e__, short __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_short* __a__, short* __e__, short __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_short* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_ushort* __a__ )
{ return false; }

inline unsigned short atomic_load_explicit
( volatile atomic_ushort* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline unsigned short atomic_load( volatile atomic_ushort* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_ushort* __a__, unsigned short __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_ushort* __a__, unsigned short __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline unsigned short atomic_swap_explicit
( volatile atomic_ushort* __a__, unsigned short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline unsigned short atomic_swap
( volatile atomic_ushort* __a__, unsigned short __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_ushort* __a__, unsigned short* __e__, unsigned short __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_ushort* __a__, unsigned short* __e__, unsigned short __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_ushort* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_int* __a__ )
{ return false; }

inline int atomic_load_explicit
( volatile atomic_int* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline int atomic_load( volatile atomic_int* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_int* __a__, int __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_int* __a__, int __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline int atomic_swap_explicit
( volatile atomic_int* __a__, int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline int atomic_swap
( volatile atomic_int* __a__, int __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_int* __a__, int* __e__, int __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_int* __a__, int* __e__, int __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_int* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_uint* __a__ )
{ return false; }

inline unsigned int atomic_load_explicit
( volatile atomic_uint* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline unsigned int atomic_load( volatile atomic_uint* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_uint* __a__, unsigned int __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_uint* __a__, unsigned int __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline unsigned int atomic_swap_explicit
( volatile atomic_uint* __a__, unsigned int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline unsigned int atomic_swap
( volatile atomic_uint* __a__, unsigned int __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_uint* __a__, unsigned int* __e__, unsigned int __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_uint* __a__, unsigned int* __e__, unsigned int __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_uint* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_long* __a__ )
{ return false; }

inline long atomic_load_explicit
( volatile atomic_long* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline long atomic_load( volatile atomic_long* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_long* __a__, long __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_long* __a__, long __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline long atomic_swap_explicit
( volatile atomic_long* __a__, long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline long atomic_swap
( volatile atomic_long* __a__, long __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_long* __a__, long* __e__, long __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_long* __a__, long* __e__, long __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_long* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_ulong* __a__ )
{ return false; }

inline unsigned long atomic_load_explicit
( volatile atomic_ulong* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline unsigned long atomic_load( volatile atomic_ulong* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_ulong* __a__, unsigned long __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_ulong* __a__, unsigned long __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline unsigned long atomic_swap_explicit
( volatile atomic_ulong* __a__, unsigned long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline unsigned long atomic_swap
( volatile atomic_ulong* __a__, unsigned long __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_ulong* __a__, unsigned long* __e__, unsigned long __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_ulong* __a__, unsigned long* __e__, unsigned long __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_ulong* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_llong* __a__ )
{ return false; }

inline long long atomic_load_explicit
( volatile atomic_llong* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline long long atomic_load( volatile atomic_llong* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_llong* __a__, long long __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_llong* __a__, long long __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline long long atomic_swap_explicit
( volatile atomic_llong* __a__, long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline long long atomic_swap
( volatile atomic_llong* __a__, long long __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_llong* __a__, long long* __e__, long long __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_llong* __a__, long long* __e__, long long __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_llong* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_ullong* __a__ )
{ return false; }

inline unsigned long long atomic_load_explicit
( volatile atomic_ullong* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline unsigned long long atomic_load( volatile atomic_ullong* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_ullong* __a__, unsigned long long __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_ullong* __a__, unsigned long long __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline unsigned long long atomic_swap_explicit
( volatile atomic_ullong* __a__, unsigned long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline unsigned long long atomic_swap
( volatile atomic_ullong* __a__, unsigned long long __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_ullong* __a__, unsigned long long* __e__, unsigned long long __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_ullong* __a__, unsigned long long* __e__, unsigned long long __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_ullong* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline bool atomic_is_lock_free( const volatile atomic_wchar_t* __a__ )
{ return false; }

inline wchar_t atomic_load_explicit
( volatile atomic_wchar_t* __a__, memory_order __x__ )
{ return _ATOMIC_LOAD_( __a__, __x__ ); }

inline wchar_t atomic_load( volatile atomic_wchar_t* __a__ )
{ return atomic_load_explicit( __a__, memory_order_seq_cst ); }

inline void atomic_store_explicit
( volatile atomic_wchar_t* __a__, wchar_t __m__, memory_order __x__ )
{ _ATOMIC_STORE_( __a__, __m__, __x__ ); }

inline void atomic_store
( volatile atomic_wchar_t* __a__, wchar_t __m__ )
{ atomic_store_explicit( __a__, __m__, memory_order_seq_cst ); }

inline wchar_t atomic_swap_explicit
( volatile atomic_wchar_t* __a__, wchar_t __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, =, __m__, __x__ ); }

inline wchar_t atomic_swap
( volatile atomic_wchar_t* __a__, wchar_t __m__ )
{ return atomic_swap_explicit( __a__, __m__, memory_order_seq_cst ); }

inline bool atomic_compare_swap_explicit
( volatile atomic_wchar_t* __a__, wchar_t* __e__, wchar_t __m__,
  memory_order __x__, memory_order __y__ )
{ return _ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ ); }

inline bool atomic_compare_swap
( volatile atomic_wchar_t* __a__, wchar_t* __e__, wchar_t __m__ )
{ return atomic_compare_swap_explicit( __a__, __e__, __m__,
                 memory_order_seq_cst, memory_order_seq_cst ); }

inline void atomic_fence
( const volatile atomic_wchar_t* __a__, memory_order __x__ )
{ _ATOMIC_FENCE_( __a__, __x__ ); }


inline void* atomic_fetch_add_explicit
( volatile atomic_address* __a__, ptrdiff_t __m__, memory_order __x__ )
{ void* volatile* __p__ = &((__a__)->__f__);
  volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ );
  __atomic_flag_wait_explicit__( __g__, __x__ );
  void* __r__ = *__p__;
  *__p__ = (void*)((char*)(*__p__) + __m__);
  atomic_flag_clear_explicit( __g__, __x__ );
  return __r__; }

inline void* atomic_fetch_add
( volatile atomic_address* __a__, ptrdiff_t __m__ )
{ return atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline void* atomic_fetch_sub_explicit
( volatile atomic_address* __a__, ptrdiff_t __m__, memory_order __x__ )
{ void* volatile* __p__ = &((__a__)->__f__);
  volatile atomic_flag* __g__ = __atomic_flag_for_address__( __p__ );
  __atomic_flag_wait_explicit__( __g__, __x__ );
  void* __r__ = *__p__;
  *__p__ = (void*)((char*)(*__p__) - __m__);
  atomic_flag_clear_explicit( __g__, __x__ );
  return __r__; }

inline void* atomic_fetch_sub
( volatile atomic_address* __a__, ptrdiff_t __m__ )
{ return atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline char atomic_fetch_add_explicit
( volatile atomic_char* __a__, char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline char atomic_fetch_add
( volatile atomic_char* __a__, char __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline char atomic_fetch_sub_explicit
( volatile atomic_char* __a__, char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline char atomic_fetch_sub
( volatile atomic_char* __a__, char __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline char atomic_fetch_and_explicit
( volatile atomic_char* __a__, char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline char atomic_fetch_and
( volatile atomic_char* __a__, char __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline char atomic_fetch_or_explicit
( volatile atomic_char* __a__, char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline char atomic_fetch_or
( volatile atomic_char* __a__, char __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline char atomic_fetch_xor_explicit
( volatile atomic_char* __a__, char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline char atomic_fetch_xor
( volatile atomic_char* __a__, char __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline signed char atomic_fetch_add_explicit
( volatile atomic_schar* __a__, signed char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline signed char atomic_fetch_add
( volatile atomic_schar* __a__, signed char __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline signed char atomic_fetch_sub_explicit
( volatile atomic_schar* __a__, signed char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline signed char atomic_fetch_sub
( volatile atomic_schar* __a__, signed char __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline signed char atomic_fetch_and_explicit
( volatile atomic_schar* __a__, signed char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline signed char atomic_fetch_and
( volatile atomic_schar* __a__, signed char __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline signed char atomic_fetch_or_explicit
( volatile atomic_schar* __a__, signed char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline signed char atomic_fetch_or
( volatile atomic_schar* __a__, signed char __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline signed char atomic_fetch_xor_explicit
( volatile atomic_schar* __a__, signed char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline signed char atomic_fetch_xor
( volatile atomic_schar* __a__, signed char __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned char atomic_fetch_add_explicit
( volatile atomic_uchar* __a__, unsigned char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline unsigned char atomic_fetch_add
( volatile atomic_uchar* __a__, unsigned char __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned char atomic_fetch_sub_explicit
( volatile atomic_uchar* __a__, unsigned char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline unsigned char atomic_fetch_sub
( volatile atomic_uchar* __a__, unsigned char __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned char atomic_fetch_and_explicit
( volatile atomic_uchar* __a__, unsigned char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline unsigned char atomic_fetch_and
( volatile atomic_uchar* __a__, unsigned char __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned char atomic_fetch_or_explicit
( volatile atomic_uchar* __a__, unsigned char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline unsigned char atomic_fetch_or
( volatile atomic_uchar* __a__, unsigned char __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned char atomic_fetch_xor_explicit
( volatile atomic_uchar* __a__, unsigned char __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline unsigned char atomic_fetch_xor
( volatile atomic_uchar* __a__, unsigned char __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline short atomic_fetch_add_explicit
( volatile atomic_short* __a__, short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline short atomic_fetch_add
( volatile atomic_short* __a__, short __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline short atomic_fetch_sub_explicit
( volatile atomic_short* __a__, short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline short atomic_fetch_sub
( volatile atomic_short* __a__, short __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline short atomic_fetch_and_explicit
( volatile atomic_short* __a__, short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline short atomic_fetch_and
( volatile atomic_short* __a__, short __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline short atomic_fetch_or_explicit
( volatile atomic_short* __a__, short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline short atomic_fetch_or
( volatile atomic_short* __a__, short __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline short atomic_fetch_xor_explicit
( volatile atomic_short* __a__, short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline short atomic_fetch_xor
( volatile atomic_short* __a__, short __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned short atomic_fetch_add_explicit
( volatile atomic_ushort* __a__, unsigned short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline unsigned short atomic_fetch_add
( volatile atomic_ushort* __a__, unsigned short __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned short atomic_fetch_sub_explicit
( volatile atomic_ushort* __a__, unsigned short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline unsigned short atomic_fetch_sub
( volatile atomic_ushort* __a__, unsigned short __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned short atomic_fetch_and_explicit
( volatile atomic_ushort* __a__, unsigned short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline unsigned short atomic_fetch_and
( volatile atomic_ushort* __a__, unsigned short __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned short atomic_fetch_or_explicit
( volatile atomic_ushort* __a__, unsigned short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline unsigned short atomic_fetch_or
( volatile atomic_ushort* __a__, unsigned short __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned short atomic_fetch_xor_explicit
( volatile atomic_ushort* __a__, unsigned short __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline unsigned short atomic_fetch_xor
( volatile atomic_ushort* __a__, unsigned short __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline int atomic_fetch_add_explicit
( volatile atomic_int* __a__, int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline int atomic_fetch_add
( volatile atomic_int* __a__, int __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline int atomic_fetch_sub_explicit
( volatile atomic_int* __a__, int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline int atomic_fetch_sub
( volatile atomic_int* __a__, int __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline int atomic_fetch_and_explicit
( volatile atomic_int* __a__, int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline int atomic_fetch_and
( volatile atomic_int* __a__, int __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline int atomic_fetch_or_explicit
( volatile atomic_int* __a__, int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline int atomic_fetch_or
( volatile atomic_int* __a__, int __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline int atomic_fetch_xor_explicit
( volatile atomic_int* __a__, int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline int atomic_fetch_xor
( volatile atomic_int* __a__, int __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned int atomic_fetch_add_explicit
( volatile atomic_uint* __a__, unsigned int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline unsigned int atomic_fetch_add
( volatile atomic_uint* __a__, unsigned int __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned int atomic_fetch_sub_explicit
( volatile atomic_uint* __a__, unsigned int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline unsigned int atomic_fetch_sub
( volatile atomic_uint* __a__, unsigned int __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned int atomic_fetch_and_explicit
( volatile atomic_uint* __a__, unsigned int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline unsigned int atomic_fetch_and
( volatile atomic_uint* __a__, unsigned int __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned int atomic_fetch_or_explicit
( volatile atomic_uint* __a__, unsigned int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline unsigned int atomic_fetch_or
( volatile atomic_uint* __a__, unsigned int __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned int atomic_fetch_xor_explicit
( volatile atomic_uint* __a__, unsigned int __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline unsigned int atomic_fetch_xor
( volatile atomic_uint* __a__, unsigned int __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long atomic_fetch_add_explicit
( volatile atomic_long* __a__, long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline long atomic_fetch_add
( volatile atomic_long* __a__, long __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long atomic_fetch_sub_explicit
( volatile atomic_long* __a__, long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline long atomic_fetch_sub
( volatile atomic_long* __a__, long __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long atomic_fetch_and_explicit
( volatile atomic_long* __a__, long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline long atomic_fetch_and
( volatile atomic_long* __a__, long __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long atomic_fetch_or_explicit
( volatile atomic_long* __a__, long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline long atomic_fetch_or
( volatile atomic_long* __a__, long __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long atomic_fetch_xor_explicit
( volatile atomic_long* __a__, long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline long atomic_fetch_xor
( volatile atomic_long* __a__, long __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long atomic_fetch_add_explicit
( volatile atomic_ulong* __a__, unsigned long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline unsigned long atomic_fetch_add
( volatile atomic_ulong* __a__, unsigned long __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long atomic_fetch_sub_explicit
( volatile atomic_ulong* __a__, unsigned long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline unsigned long atomic_fetch_sub
( volatile atomic_ulong* __a__, unsigned long __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long atomic_fetch_and_explicit
( volatile atomic_ulong* __a__, unsigned long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline unsigned long atomic_fetch_and
( volatile atomic_ulong* __a__, unsigned long __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long atomic_fetch_or_explicit
( volatile atomic_ulong* __a__, unsigned long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline unsigned long atomic_fetch_or
( volatile atomic_ulong* __a__, unsigned long __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long atomic_fetch_xor_explicit
( volatile atomic_ulong* __a__, unsigned long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline unsigned long atomic_fetch_xor
( volatile atomic_ulong* __a__, unsigned long __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long long atomic_fetch_add_explicit
( volatile atomic_llong* __a__, long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline long long atomic_fetch_add
( volatile atomic_llong* __a__, long long __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long long atomic_fetch_sub_explicit
( volatile atomic_llong* __a__, long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline long long atomic_fetch_sub
( volatile atomic_llong* __a__, long long __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long long atomic_fetch_and_explicit
( volatile atomic_llong* __a__, long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline long long atomic_fetch_and
( volatile atomic_llong* __a__, long long __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long long atomic_fetch_or_explicit
( volatile atomic_llong* __a__, long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline long long atomic_fetch_or
( volatile atomic_llong* __a__, long long __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline long long atomic_fetch_xor_explicit
( volatile atomic_llong* __a__, long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline long long atomic_fetch_xor
( volatile atomic_llong* __a__, long long __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long long atomic_fetch_add_explicit
( volatile atomic_ullong* __a__, unsigned long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline unsigned long long atomic_fetch_add
( volatile atomic_ullong* __a__, unsigned long long __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long long atomic_fetch_sub_explicit
( volatile atomic_ullong* __a__, unsigned long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline unsigned long long atomic_fetch_sub
( volatile atomic_ullong* __a__, unsigned long long __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long long atomic_fetch_and_explicit
( volatile atomic_ullong* __a__, unsigned long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline unsigned long long atomic_fetch_and
( volatile atomic_ullong* __a__, unsigned long long __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long long atomic_fetch_or_explicit
( volatile atomic_ullong* __a__, unsigned long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline unsigned long long atomic_fetch_or
( volatile atomic_ullong* __a__, unsigned long long __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline unsigned long long atomic_fetch_xor_explicit
( volatile atomic_ullong* __a__, unsigned long long __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline unsigned long long atomic_fetch_xor
( volatile atomic_ullong* __a__, unsigned long long __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


inline wchar_t atomic_fetch_add_explicit
( volatile atomic_wchar_t* __a__, wchar_t __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, +=, __m__, __x__ ); }

inline wchar_t atomic_fetch_add
( volatile atomic_wchar_t* __a__, wchar_t __m__ )
{ atomic_fetch_add_explicit( __a__, __m__, memory_order_seq_cst ); }


inline wchar_t atomic_fetch_sub_explicit
( volatile atomic_wchar_t* __a__, wchar_t __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, -=, __m__, __x__ ); }

inline wchar_t atomic_fetch_sub
( volatile atomic_wchar_t* __a__, wchar_t __m__ )
{ atomic_fetch_sub_explicit( __a__, __m__, memory_order_seq_cst ); }


inline wchar_t atomic_fetch_and_explicit
( volatile atomic_wchar_t* __a__, wchar_t __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, &=, __m__, __x__ ); }

inline wchar_t atomic_fetch_and
( volatile atomic_wchar_t* __a__, wchar_t __m__ )
{ atomic_fetch_and_explicit( __a__, __m__, memory_order_seq_cst ); }


inline wchar_t atomic_fetch_or_explicit
( volatile atomic_wchar_t* __a__, wchar_t __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, |=, __m__, __x__ ); }

inline wchar_t atomic_fetch_or
( volatile atomic_wchar_t* __a__, wchar_t __m__ )
{ atomic_fetch_or_explicit( __a__, __m__, memory_order_seq_cst ); }


inline wchar_t atomic_fetch_xor_explicit
( volatile atomic_wchar_t* __a__, wchar_t __m__, memory_order __x__ )
{ return _ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ ); }

inline wchar_t atomic_fetch_xor
( volatile atomic_wchar_t* __a__, wchar_t __m__ )
{ atomic_fetch_xor_explicit( __a__, __m__, memory_order_seq_cst ); }


#else


#define atomic_is_lock_free( __a__ ) \
false

#define atomic_load( __a__ ) \
_ATOMIC_LOAD_( __a__, memory_order_seq_cst )

#define atomic_load_explicit( __a__, __x__ ) \
_ATOMIC_LOAD_( __a__, __x__ )

#define atomic_store( __a__, __m__ ) \
_ATOMIC_STORE_( __a__, __m__, memory_order_seq_cst )

#define atomic_store_explicit( __a__, __m__, __x__ ) \
_ATOMIC_STORE_( __a__, __m__, __x__ )

#define atomic_swap( __a__, __m__ ) \
_ATOMIC_MODIFY_( __a__, =, __m__, memory_order_seq_cst )

#define atomic_swap_explicit( __a__, __m__, __x__ ) \
_ATOMIC_MODIFY_( __a__, =, __m__, __x__ )

#define atomic_compare_swap( __a__, __e__, __m__ ) \
_ATOMIC_CMPSWP_( __a__, __e__, __m__, memory_order_seq_cst )

#define atomic_compare_swap_explicit( __a__, __e__, __m__, __x__, __y__ ) \
_ATOMIC_CMPSWP_( __a__, __e__, __m__, __x__ )

#define atomic_fence( __a__, __x__ ) \
({ _ATOMIC_FENCE_( __a__, __x__ ); })


#define atomic_fetch_add_explicit( __a__, __m__, __x__ ) \
_ATOMIC_MODIFY_( __a__, +=, __m__, __x__ )

#define atomic_fetch_add( __a__, __m__ ) \
_ATOMIC_MODIFY_( __a__, +=, __m__, memory_order_seq_cst )


#define atomic_fetch_sub_explicit( __a__, __m__, __x__ ) \
_ATOMIC_MODIFY_( __a__, -=, __m__, __x__ )

#define atomic_fetch_sub( __a__, __m__ ) \
_ATOMIC_MODIFY_( __a__, -=, __m__, memory_order_seq_cst )


#define atomic_fetch_and_explicit( __a__, __m__, __x__ ) \
_ATOMIC_MODIFY_( __a__, &=, __m__, __x__ )

#define atomic_fetch_and( __a__, __m__ ) \
_ATOMIC_MODIFY_( __a__, &=, __m__, memory_order_seq_cst )


#define atomic_fetch_or_explicit( __a__, __m__, __x__ ) \
_ATOMIC_MODIFY_( __a__, |=, __m__, __x__ )

#define atomic_fetch_or( __a__, __m__ ) \
_ATOMIC_MODIFY_( __a__, |=, __m__, memory_order_seq_cst )


#define atomic_fetch_xor_explicit( __a__, __m__, __x__ ) \
_ATOMIC_MODIFY_( __a__, ^=, __m__, __x__ )

#define atomic_fetch_xor( __a__, __m__ ) \
_ATOMIC_MODIFY_( __a__, ^=, __m__, memory_order_seq_cst )


#endif


#ifdef __cplusplus


inline bool atomic_bool::is_lock_free() const volatile
{ return false; }

inline void atomic_bool::store
( bool __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline bool atomic_bool::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline bool atomic_bool::swap
( bool __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_bool::compare_swap
( bool& __e__, bool __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_bool::compare_swap
( bool& __e__, bool __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_bool::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_address::is_lock_free() const volatile
{ return false; }

inline void atomic_address::store
( void* __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline void* atomic_address::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline void* atomic_address::swap
( void* __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_address::compare_swap
( void*& __e__, void* __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_address::compare_swap
( void*& __e__, void* __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_address::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_char::is_lock_free() const volatile
{ return false; }

inline void atomic_char::store
( char __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline char atomic_char::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline char atomic_char::swap
( char __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_char::compare_swap
( char& __e__, char __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_char::compare_swap
( char& __e__, char __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_char::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_schar::is_lock_free() const volatile
{ return false; }

inline void atomic_schar::store
( signed char __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline signed char atomic_schar::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline signed char atomic_schar::swap
( signed char __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_schar::compare_swap
( signed char& __e__, signed char __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_schar::compare_swap
( signed char& __e__, signed char __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_schar::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_uchar::is_lock_free() const volatile
{ return false; }

inline void atomic_uchar::store
( unsigned char __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline unsigned char atomic_uchar::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline unsigned char atomic_uchar::swap
( unsigned char __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_uchar::compare_swap
( unsigned char& __e__, unsigned char __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_uchar::compare_swap
( unsigned char& __e__, unsigned char __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_uchar::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_short::is_lock_free() const volatile
{ return false; }

inline void atomic_short::store
( short __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline short atomic_short::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline short atomic_short::swap
( short __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_short::compare_swap
( short& __e__, short __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_short::compare_swap
( short& __e__, short __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_short::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_ushort::is_lock_free() const volatile
{ return false; }

inline void atomic_ushort::store
( unsigned short __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline unsigned short atomic_ushort::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline unsigned short atomic_ushort::swap
( unsigned short __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_ushort::compare_swap
( unsigned short& __e__, unsigned short __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_ushort::compare_swap
( unsigned short& __e__, unsigned short __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_ushort::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_int::is_lock_free() const volatile
{ return false; }

inline void atomic_int::store
( int __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline int atomic_int::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline int atomic_int::swap
( int __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_int::compare_swap
( int& __e__, int __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_int::compare_swap
( int& __e__, int __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_int::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_uint::is_lock_free() const volatile
{ return false; }

inline void atomic_uint::store
( unsigned int __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline unsigned int atomic_uint::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline unsigned int atomic_uint::swap
( unsigned int __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_uint::compare_swap
( unsigned int& __e__, unsigned int __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_uint::compare_swap
( unsigned int& __e__, unsigned int __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_uint::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_long::is_lock_free() const volatile
{ return false; }

inline void atomic_long::store
( long __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline long atomic_long::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline long atomic_long::swap
( long __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_long::compare_swap
( long& __e__, long __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_long::compare_swap
( long& __e__, long __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_long::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_ulong::is_lock_free() const volatile
{ return false; }

inline void atomic_ulong::store
( unsigned long __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline unsigned long atomic_ulong::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline unsigned long atomic_ulong::swap
( unsigned long __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_ulong::compare_swap
( unsigned long& __e__, unsigned long __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_ulong::compare_swap
( unsigned long& __e__, unsigned long __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_ulong::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_llong::is_lock_free() const volatile
{ return false; }

inline void atomic_llong::store
( long long __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline long long atomic_llong::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline long long atomic_llong::swap
( long long __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_llong::compare_swap
( long long& __e__, long long __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_llong::compare_swap
( long long& __e__, long long __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_llong::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_ullong::is_lock_free() const volatile
{ return false; }

inline void atomic_ullong::store
( unsigned long long __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline unsigned long long atomic_ullong::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline unsigned long long atomic_ullong::swap
( unsigned long long __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_ullong::compare_swap
( unsigned long long& __e__, unsigned long long __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_ullong::compare_swap
( unsigned long long& __e__, unsigned long long __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_ullong::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


inline bool atomic_wchar_t::is_lock_free() const volatile
{ return false; }

inline void atomic_wchar_t::store
( wchar_t __m__, memory_order __x__ ) volatile
{ atomic_store_explicit( this, __m__, __x__ ); }

inline wchar_t atomic_wchar_t::load
( memory_order __x__ ) volatile
{ return atomic_load_explicit( this, __x__ ); }

inline wchar_t atomic_wchar_t::swap
( wchar_t __m__, memory_order __x__ ) volatile
{ return atomic_swap_explicit( this, __m__, __x__ ); }

inline bool atomic_wchar_t::compare_swap
( wchar_t& __e__, wchar_t __m__,
  memory_order __x__, memory_order __y__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__, __y__ ); }

inline bool atomic_wchar_t::compare_swap
( wchar_t& __e__, wchar_t __m__, memory_order __x__ ) volatile
{ return atomic_compare_swap_explicit( this, &__e__, __m__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

inline void atomic_wchar_t::fence
( memory_order __x__ ) const volatile
{ return atomic_fence( this, __x__ ); }


template< typename T >
inline bool atomic<T>::is_lock_free() const volatile
{ return false; }

template< typename T >
inline void atomic<T>::store( T __v__, memory_order __x__ ) volatile
{ _ATOMIC_STORE_( this, __v__, __x__ ); }

template< typename T >
inline T atomic<T>::load( memory_order __x__ ) volatile
{ return _ATOMIC_LOAD_( this, __x__ ); }

template< typename T >
inline T atomic<T>::swap( T __v__, memory_order __x__ ) volatile
{ return _ATOMIC_MODIFY_( this, =, __v__, __x__ ); }

template< typename T >
inline bool atomic<T>::compare_swap
( T& __r__, T __v__, memory_order __x__, memory_order __y__ ) volatile
{ return _ATOMIC_CMPSWP_( this, &__r__, __v__, __x__ ); }

template< typename T >
inline bool atomic<T>::compare_swap
( T& __r__, T __v__, memory_order __x__ ) volatile
{ return compare_swap( __r__, __v__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }


inline void* atomic_address::fetch_add
( ptrdiff_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }

inline void* atomic_address::fetch_sub
( ptrdiff_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline char atomic_char::fetch_add
( char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline char atomic_char::fetch_sub
( char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline char atomic_char::fetch_and
( char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline char atomic_char::fetch_or
( char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline char atomic_char::fetch_xor
( char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline signed char atomic_schar::fetch_add
( signed char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline signed char atomic_schar::fetch_sub
( signed char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline signed char atomic_schar::fetch_and
( signed char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline signed char atomic_schar::fetch_or
( signed char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline signed char atomic_schar::fetch_xor
( signed char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline unsigned char atomic_uchar::fetch_add
( unsigned char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline unsigned char atomic_uchar::fetch_sub
( unsigned char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline unsigned char atomic_uchar::fetch_and
( unsigned char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline unsigned char atomic_uchar::fetch_or
( unsigned char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline unsigned char atomic_uchar::fetch_xor
( unsigned char __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline short atomic_short::fetch_add
( short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline short atomic_short::fetch_sub
( short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline short atomic_short::fetch_and
( short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline short atomic_short::fetch_or
( short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline short atomic_short::fetch_xor
( short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline unsigned short atomic_ushort::fetch_add
( unsigned short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline unsigned short atomic_ushort::fetch_sub
( unsigned short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline unsigned short atomic_ushort::fetch_and
( unsigned short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline unsigned short atomic_ushort::fetch_or
( unsigned short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline unsigned short atomic_ushort::fetch_xor
( unsigned short __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline int atomic_int::fetch_add
( int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline int atomic_int::fetch_sub
( int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline int atomic_int::fetch_and
( int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline int atomic_int::fetch_or
( int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline int atomic_int::fetch_xor
( int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline unsigned int atomic_uint::fetch_add
( unsigned int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline unsigned int atomic_uint::fetch_sub
( unsigned int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline unsigned int atomic_uint::fetch_and
( unsigned int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline unsigned int atomic_uint::fetch_or
( unsigned int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline unsigned int atomic_uint::fetch_xor
( unsigned int __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline long atomic_long::fetch_add
( long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline long atomic_long::fetch_sub
( long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline long atomic_long::fetch_and
( long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline long atomic_long::fetch_or
( long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline long atomic_long::fetch_xor
( long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline unsigned long atomic_ulong::fetch_add
( unsigned long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline unsigned long atomic_ulong::fetch_sub
( unsigned long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline unsigned long atomic_ulong::fetch_and
( unsigned long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline unsigned long atomic_ulong::fetch_or
( unsigned long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline unsigned long atomic_ulong::fetch_xor
( unsigned long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline long long atomic_llong::fetch_add
( long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline long long atomic_llong::fetch_sub
( long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline long long atomic_llong::fetch_and
( long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline long long atomic_llong::fetch_or
( long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline long long atomic_llong::fetch_xor
( long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline unsigned long long atomic_ullong::fetch_add
( unsigned long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline unsigned long long atomic_ullong::fetch_sub
( unsigned long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline unsigned long long atomic_ullong::fetch_and
( unsigned long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline unsigned long long atomic_ullong::fetch_or
( unsigned long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline unsigned long long atomic_ullong::fetch_xor
( unsigned long long __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


inline wchar_t atomic_wchar_t::fetch_add
( wchar_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, __m__, __x__ ); }


inline wchar_t atomic_wchar_t::fetch_sub
( wchar_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, __m__, __x__ ); }


inline wchar_t atomic_wchar_t::fetch_and
( wchar_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_and_explicit( this, __m__, __x__ ); }


inline wchar_t atomic_wchar_t::fetch_or
( wchar_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_or_explicit( this, __m__, __x__ ); }


inline wchar_t atomic_wchar_t::fetch_xor
( wchar_t __m__, memory_order __x__ ) volatile
{ return atomic_fetch_xor_explicit( this, __m__, __x__ ); }


template< typename T >
T* atomic<T*>::load( memory_order __x__ ) volatile
{ return static_cast<T*>( atomic_address::load( __x__ ) ); }

template< typename T >
T* atomic<T*>::swap( T* __v__, memory_order __x__ ) volatile
{ return static_cast<T*>( atomic_address::swap( __v__, __x__ ) ); }

template< typename T >
bool atomic<T*>::compare_swap
( T*& __r__, T* __v__, memory_order __x__, memory_order __y__) volatile
{ return atomic_address::compare_swap( *reinterpret_cast<void**>( &__r__ ),
               static_cast<void*>( __v__ ), __x__, __y__ ); }
//{ return _ATOMIC_CMPSWP_( this, &__r__, __v__, __x__ ); }

template< typename T >
bool atomic<T*>::compare_swap
( T*& __r__, T* __v__, memory_order __x__ ) volatile
{ return compare_swap( __r__, __v__, __x__,
      __x__ == memory_order_acq_rel ? memory_order_acquire :
      __x__ == memory_order_release ? memory_order_relaxed : __x__ ); }

template< typename T >
T* atomic<T*>::fetch_add( ptrdiff_t __v__, memory_order __x__ ) volatile
{ return atomic_fetch_add_explicit( this, sizeof(T) * __v__, __x__ ); }

template< typename T >
T* atomic<T*>::fetch_sub( ptrdiff_t __v__, memory_order __x__ ) volatile
{ return atomic_fetch_sub_explicit( this, sizeof(T) * __v__, __x__ ); }


#endif


#ifdef __cplusplus
} // namespace std
#endif

