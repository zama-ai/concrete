#include <concretelang/ServerLib/ServerLambda.h>

void answer_client(MyConnection conn) {
    std::istream from_client = conn.istream();
    std::ostream to_client = conn.ostream();
    auto err = serverLambda.read_call_write(serverInput, serverOutput);
    if (err) {
        throw MyException();
    }
}
